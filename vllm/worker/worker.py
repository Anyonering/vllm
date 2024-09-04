"""A GPU worker class."""
import gc
import os
from typing import List, Optional, Set, Tuple, Type

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, WorkerInput


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.prompt_adapter_config = prompt_adapter_config
        self.is_driver_worker = is_driver_worker
        self.move_out_cache = []
        self.move_in_cache = []
        if parallel_config and is_driver_worker:
            assert rank % parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.multimodal_config = multimodal_config

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif self.model_config.embedding_mode:
            ModelRunnerClass = EmbeddingModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            multimodal_config=multimodal_config,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.tensor]]] = None
        # the stream pool that are available for use
        self.cache_stream_pool = [torch.cuda.Stream() for i in range(cache_config.num_stream)]
        # the stream pool that are currently in use
        self.stream_in_use = []
        self.move_out_counter = 0

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        blocks_to_refill = torch.tensor(execute_model_req.blocks_to_refill,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_kick_out = torch.tensor(execute_model_req.blocks_to_kick_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        refill_index = torch.tensor(execute_model_req.refill_index, device="cpu", dtype=torch.int64)
        kick_out_index = torch.tensor(execute_model_req.kick_out_index, device="cpu", dtype=torch.int64)
        refill_stream = torch.tensor(execute_model_req.refill_stream, device="cpu", dtype=torch.int64)
        kick_out_stream = torch.tensor(execute_model_req.kick_out_stream, device="cpu", dtype=torch.int64)
        stream_to_sync = torch.tensor(execute_model_req.stream_to_sync, device="cpu", dtype=torch.int64)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_refill=blocks_to_refill,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_kick_out=blocks_to_kick_out,
            kick_out_index=kick_out_index,
            refill_index=refill_index,
            refill_stream=refill_stream,
            kick_out_stream=kick_out_stream,
            stream_to_sync=stream_to_sync,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            # print(worker_input.blocks_to_swap_out)
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)
            
    @torch.inference_mode()
    def execute_worker_cache(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_kick_out is not None
                and worker_input.blocks_to_kick_out.numel() > 0):
            # print("blocks_to_kick_out",worker_input.blocks_to_kick_out)
            start_pos = 0
            # print(worker_input.kick_out_index.size(dim=0))
            for i in range(worker_input.kick_out_index.size(dim=0)):
                # self.move_out_counter+=1
                # print("move out counter: ", self.move_out_counter)
                # if(self.move_out_counter == 2 or self.move_out_counter == 4):
                #     list_indices = worker_input.blocks_to_kick_out[start_pos:worker_input.kick_out_index[i]][:,0]
                #     print("move out list indices: ",list_indices)
                #     for j in range(self.cache_engine[virtual_engine].num_attention_layers):
                #         # print("gpu caches shape: ",self.gpu_cache[0][j].shape)
                #         self.move_out_cache.append(self.gpu_cache[0][j][:,list_indices,:].clone().detach()) 
                #     # print("mvoe out cache: ",self.move_out_cache[0])
                #     cuda = torch.device('cuda')
                #     self.cache_stream_pool[worker_input.kick_out_stream[i]].wait_stream(torch.cuda.default_stream(cuda))
                with torch.cuda.stream(self.cache_stream_pool[worker_input.kick_out_stream[i]]):
                    # print("kick out stream index: ",worker_input.kick_out_stream[i])
                    self.cache_engine[virtual_engine].swap_out(
                        worker_input.blocks_to_kick_out[start_pos:worker_input.kick_out_index[i]])
                    # print("kicking blocks table: ",worker_input.blocks_to_kick_out[start_pos:worker_input.kick_out_index[i]])
                start_pos += worker_input.kick_out_index[i]
        
        if (worker_input.blocks_to_refill is not None
                and worker_input.blocks_to_refill.numel() > 0):
            print("blocks_to_refill",worker_input.blocks_to_refill)
            print("worker input refill stream: ",worker_input.refill_stream)
            print("refill index: ",worker_input.refill_index)
            start_pos = 0
            for i in range(worker_input.refill_index.size(dim=0)):
                with torch.cuda.stream(self.cache_stream_pool[worker_input.refill_stream[i]]):
                    # print("refill stream index: ",worker_input.refill_stream[i])
                    self.cache_engine[virtual_engine].swap_in(
                        worker_input.blocks_to_refill[start_pos:start_pos+worker_input.refill_index[i]])
                    print("refilling blocks table: ",worker_input.blocks_to_refill[start_pos:start_pos+worker_input.refill_index[i]])
                start_pos += worker_input.refill_index[i]
                # self.cache_stream_pool[worker_input.refill_stream[i]].synchronize()
                # if(len(self.move_in_cache)==0):
                #     my_folder = "~/personal/projects/vllm_inference/kv_cache_ref"
                #     list_indices = worker_input.blocks_to_refill[0:worker_input.refill_index[i]][:,1]
                #     print("move in list indices: ",list_indices)
                #     for j in range(self.cache_engine[virtual_engine].num_attention_layers):
                #         self.move_in_cache.append(self.gpu_cache[0][j][:,list_indices,:].clone().detach())
                #     for j in range(self.cache_engine[virtual_engine].num_attention_layers):
                #         print(f"In layer {j}",torch.all(torch.eq(self.move_out_cache[j],self.move_in_cache[j])))
                #         torch.save(self.move_out_cache[j],f"{my_folder}/worker_cache_{j}.pt")
        if(worker_input.stream_to_sync is not None and worker_input.stream_to_sync.numel() > 0):
            print("stream_to_sync: ",worker_input.stream_to_sync)
            for i in worker_input.stream_to_sync:
                self.cache_stream_pool[i].synchronize()
            
        # if(worker_input.blocks_to_kick_out is not None
        #         and worker_input.blocks_to_kick_out.numel() > 0 and
        #         worker_input.blocks_to_refill is not None
        #         and worker_input.blocks_to_refill.numel() > 0):
        #     print("Warning: this is dangerous to kick out and refill at the same time!\nWarning: this is dangerous to kick out and refill at the same time!\n")
        
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
