import enum
import os
import random
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class ComparableEnum(enum.Enum):
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class BlockStatus(ComparableEnum):
    NOT_ALLOCATED = -1
    RUNNING = 0
    KICKING_OUT = 1
    ON_HOST = 2
    SWAPPING_IN = 3
    ON_DEVICE = 4
    
 
class RequestStatus(ComparableEnum):
    NEW_ARRIVAL = 0
    REQ_SCHEDULED = 1
    TEMP_FINISHED = 2
    INFO_REQ_ARR = 3
    CHAT_REQ_ARR = 4
    PERMANENT_FINISHED = 5
    
@dataclass
class SessionStatus:
    session_id: int
    req_status: RequestStatus
    block_status: BlockStatus
    is_last_round: bool
    seq_group: SequenceGroup
    current_stream: int
    
    def __init__(self, session_id: int, seq_group: SequenceGroup):
        self.session_id = session_id
        self.seq_group = seq_group
        self.req_status = RequestStatus.NEW_ARRIVAL
        self.block_status = BlockStatus.NOT_ALLOCATED
        self.current_stream = -1
        self.is_last_round = False
        
  
class SwapStatus(enum.Enum):
    SWAP_IN = enum.auto()
    SWAP_OUT = enum.auto()
    SYNCHRONIZED = enum.auto()
    
        
@dataclass   
class SetSessionBlockStatus:
    session_id: int
    block_status: BlockStatus  
    def __str__(self):
        return f"set session {self.session_id} to {self.block_status}"

@dataclass   
class KvCacheSwapMeta:   
    swap_status: SwapStatus
    mapping:  List[Tuple[int, int]]
    stream_position: List[int]
    stream_list: List[int]
    sync_this_time: bool
    #set_session: List[SetSessionBlockStatus]
    
    def __init__(self,swap_status:SwapStatus):
        self.swap_status = swap_status
        self.mapping = []
        self.stream_list = []
        self.stream_position = []
        self.sync_this_time = False
        #self.set_session = []
    def __str__(self):
        return f"swap_status: {self.swap_status}\nstream_list: {self.stream_list}"

    

class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: Iterable[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # the index of blocks to kick out
    # in the form of [len(block of req_1), len(block of req_2),...]
    kick_out_index: List[int]
    # the stream id in range(0,cache_config.num_stream) for each req
    kick_out_stream: List[int]
    # the index of blocks to refill
    # follow the same form as kick out index list
    refill_index: List[int]
    # the stream id in range(0,cache_config.num_stream) for each req
    refill_stream: List[int]
    # the stream id in range(0,cache_config.num_stream) for workers to synchronize
    stream_to_sync: List[int]
    # Blocks to refill when new user prompt may come. List of CPU -> GPU block number.
    blocks_to_refill: List[Tuple[int, int]] 
    # Blocks to swap out after a request finished. List of GPU -> CPU block number.
    blocks_to_kick_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[SequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )

@dataclass
class SeqGroupBlockInfo:
    seq_group: SequenceGroup
    num_blocks: int
    last_stream_use: int
    # block state can only be KIKCING, REFILLING, ON_DEVICE
    block_state: int
    
@dataclass
class SeqGroupWCounter:
    seq_group: SequenceGroup
    counter: int


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # a dictionary that keeps track of general info about a client session
        self.session_dict: Dict[int, SessionStatus] = {}
        # sequence group that already have their context history loading in DRAM
        # only need to allocate some blocks for new seqs tokens
        self.ready_refill: Deque[SequenceGroup] = deque()
        # sequence group that do not have their context history in DRAM but chat req has arrived
        # need to allocate both context and new seq tokens
        self.wait_refill: Deque[SequenceGroup] = deque()
        # a deque to handle kv cache swapping
        self.kv_swap_meta: Deque[KvCacheSwapMeta] = deque()
        self.current_swap_status: SwapStatus = SwapStatus.SYNCHRONIZED
        # Sequence groups temporarily finished.
        # Contain decode requests that are temporarily holding after finished.
        self.temp_finished: Deque[SequenceGroup] = deque()
        # a list of sequence waiting to be freed
        self.wait_for_free : List[Sequence] = list()
        # This contains the session_id for the group waiting to be 
        # reloaded from the host memory to the device.
        # Temporarily configured by the user 
        # by changing llm.llm_engine.scheduler[0].refill_requests
        self.refill_requests: Deque[int] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        self._finished_requests_ids: List[str] = list()
        # cuda stream that are availabe to use
        self.stream_avail = [i for i in range(0,cache_config.num_stream)]
        # cuda streams that are currently being used in workers
        self.stream_pend_sync = []
        self.stream_to_session: Dict[int, SetSessionBlockStatus] = {}
        self.trigger_sync_threshold = 3
        self.pause_load_kv = False
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step 
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode
        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0
        self.max_model_len = 2048
        self.cumulative_session_id = -1
        self.use_truncation = True
        self.is_finish_dict = {}
        self.time_in_scheduler = 0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup, session_id: int) -> None:
        is_refill = False
        seq_group.session_id = session_id
        if(self.is_finish_dict[session_id] == True):
            seq_group.is_last_round = True
        if(self.session_dict.get(session_id) is not None):
            is_refill = True
            # this session has been tracked
            # need to tell if it is ready or need kv cache reloading
            session_info = self.session_dict.get(session_id) 
            temp_block_status = session_info.block_status
            assert temp_block_status > BlockStatus.KICKING_OUT
            assert session_info.req_status >=RequestStatus.TEMP_FINISHED
            if(temp_block_status == BlockStatus.ON_DEVICE):
                # kv cache is loading, only need to allocate new seq tokens
                self.ready_refill.append(seq_group)
                session_info.req_status = RequestStatus.CHAT_REQ_ARR
                session_info.is_last_round = seq_group.is_last_round
            else:
                # kv cache is still on Host RAM.
                # need further processing
                assert temp_block_status == BlockStatus.ON_HOST or temp_block_status == BlockStatus.SWAPPING_IN
                self.wait_refill.append(seq_group)
                session_info.is_last_round = seq_group.is_last_round
                session_info.req_status = RequestStatus.CHAT_REQ_ARR
        else:
            # this is a new session
            if(session_id > self.cumulative_session_id):
                    # the given session id is greater than cumulative session id
                self.cumulative_session_id = session_id
            self.waiting.append(seq_group)
            self.session_dict[session_id] = SessionStatus(session_id=session_id,seq_group=seq_group)
            self.session_dict[session_id].is_last_round = seq_group.is_last_round
            
        print(f"adding seq group with session id{session_id}, refill: {is_refill}")
        # print(f"refill info dict has length: {len(self.refill_info_dict)} , ",self.refill_info_dict)
            
    def get_new_session_id(self, request_num: Optional[int] = None):
        if(request_num is None or request_num == 1):
            self.cumulative_session_id += 1
            print("new session id: ",self.cumulative_session_id)
            return self.cumulative_session_id
        else:
            new_session_id_list = list(range(self.cumulative_session_id+1,self.cumulative_session_id+1+request_num))
            self.cumulative_session_id +=request_num
            print("new session ids list: ",new_session_id_list)
            return new_session_id_list

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0 or len(self.ready_refill) !=0 or len(self.wait_refill) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) +len(self.ready_refill) + len(self.wait_refill)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids
    #
    # this function needs to be called from async_server
    def sync_to_host(self, session_id) -> None:
        # for sess_id in self.refill_requests:
        this_session_info = self.session_dict[session_id]
        print(f"This session {session_id} has block status: {this_session_info.block_status}")
        if(not this_session_info.block_status >=BlockStatus.KICKING_OUT):
            print(f"self.temp_finsished: {self.temp_finished}")
            print(f"self.running: {len(self.running)}")
            print(f"self.kv_swap_meta: {self.kv_swap_meta}")
            print(f"self.stream_to_session: {self.stream_to_session}")
            print(f"context group output len: {this_session_info.seq_group.get_seqs()[0].data.get_output_len()}")
            print(f"previous finished time: {this_session_info.seq_group.metrics.finished_time}, current time: {time.time()}")
            # the seq group should exist in self.temp_finished
            finished_seq_group = None
            assert this_session_info.block_status ==BlockStatus.RUNNING
            for seq_group in self.temp_finished:
                if seq_group.session_id == session_id:
                    finished_seq_group = seq_group
                    break
            assert finished_seq_group != None
            self.temp_finished.remove(finished_seq_group)
            this_session_info.req_status = RequestStatus.INFO_REQ_ARR
            this_session_info.block_status = BlockStatus.ON_DEVICE
            assert this_session_info.seq_group == finished_seq_group
            return
            
            
            
        # print(f"self.temp_finished:{self.temp_finished}")
        
        assert this_session_info.block_status >=BlockStatus.KICKING_OUT
        if(this_session_info.block_status == BlockStatus.KICKING_OUT):
            assert this_session_info.current_stream != -1
            if len(self.kv_swap_meta)> 0:
                if(self.kv_swap_meta[-1].swap_status != SwapStatus.SYNCHRONIZED):
                    self.kv_swap_meta[-1].sync_this_time = True
            else:
                self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
            #self.current_sync.append(this_session_info.last_stream_use)
            this_session_info.req_status = RequestStatus.INFO_REQ_ARR
        
    # store kv cache to host memory
    def try_store_kv_cache(self) -> None:
        if(len(self.kv_swap_meta)> 0 and self.kv_swap_meta[-1].swap_status==SwapStatus.SWAP_IN):
            # could not handle this case
            # need to sync first
            if len(self.temp_finished) >= self.trigger_sync_threshold:
                self.kv_swap_meta[-1].sync_this_time = True
                # prioritize store kv cache to host memory
                self.pause_load_kv = True
            return
        # print("getting here line 606")
        self.pause_load_kv = False
        if(len(self.temp_finished)> 0):
            # print("getting here line 610")
            # print(f"current swap status: {self.current_swap_status}")
            if(self.current_swap_status == SwapStatus.SWAP_IN):
                # print("getting here line 613")
                # print(f"self.current_swap_status: {self.current_swap_status}")
                # print(f"{self.current_swap_status == SwapStatus.SWAP_IN}")
                if len(self.kv_swap_meta)> 0:
                    self.kv_swap_meta[-1].sync_this_time = True
                else:
                    self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                return
            # print("getting here line 616")
            # only proceed when self.kv_swap_meta contains nothing or only swap_out meta
            not_proceed = any(kv_swap_meta.swap_status == SwapStatus.SWAP_IN for kv_swap_meta in self.kv_swap_meta)
            if(not_proceed):
                return 
            # print("getting here line 617")
            new_kv_swap_meta = KvCacheSwapMeta(SwapStatus.SWAP_OUT)
            while len(self.temp_finished) > 0:
                seq_group = self.temp_finished[0]
                session_id = self.temp_finished[0].session_id
                this_session_info = self.session_dict[session_id]
                # print("getting here at 624")
                assert this_session_info.block_status == BlockStatus.RUNNING
                alloc_status = self.block_manager.can_swap_finished(seq_group)
                assert alloc_status != AllocStatus.NEVER
                if(alloc_status == AllocStatus.LATER):
                    break
                else:
                    assert alloc_status == AllocStatus.OK
                    mapping = self.block_manager.swap_out_finished(seq_group=seq_group)
                    stream_id = self.stream_avail.pop(0)
                    stream_position = len(mapping)
                    # need to extend the kvcacheSwapMeta
                    new_kv_swap_meta.mapping.extend(mapping)
                    new_kv_swap_meta.stream_list.append(stream_id)
                    new_kv_swap_meta.stream_position.append(stream_position)
                    #update session_dict
                    this_session_info.current_stream = stream_id
                    this_session_info.block_status = BlockStatus.KICKING_OUT
                    # print(f"\n\n\nsetting session {session_id} to kikcing out\n\n\n")
                    self.stream_to_session[stream_id] = SetSessionBlockStatus(session_id=session_id,block_status=BlockStatus.ON_HOST)
                    # print(f"setting stream {stream_id} maps to action: {session_id} to BlockStatus.ON_HOST")
                    self.temp_finished.popleft()
            if(len(new_kv_swap_meta.stream_list)> 0 ):
                # append this meta to the deque
                self.kv_swap_meta.append(new_kv_swap_meta)
                    
    # load kv cache to device memory
    def try_load_kv_cache(self) -> None:
        # need to make sure that previous cache load is sync or is swap_in
        # otherwise might lead to cudamemory error
        if(len(self.kv_swap_meta)> 0 and self.kv_swap_meta[-1].swap_status==SwapStatus.SWAP_OUT): 
            # could not handle this case
            # need to sync first
            return
        # try to load kv cache of wait refill request
        if(len(self.wait_refill)> 0):
            if(self.current_swap_status == SwapStatus.SWAP_OUT):
                # if there are some refill requests waiting
                # issue a sync first
                if len(self.kv_swap_meta)> 0:
                    self.kv_swap_meta[-1].sync_this_time = True
                else:
                    self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                return
                # if len(self.kv_swap_meta)> 0:
                #     if self.kv_swap_meta[-1].swap_status != SwapStatus.SYNCHRONIZED and self.kv_swap_meta[-1].sync_this_time == False:
                #         self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                #     else:
                #         return
                # else:
                #     self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                # return
            
            not_proceed = any(kv_swap_meta.swap_status == SwapStatus.SWAP_OUT for kv_swap_meta in self.kv_swap_meta)
            if(not_proceed):
                return
            new_kv_swap_meta = KvCacheSwapMeta(SwapStatus.SWAP_IN)
            for refill_seq_group in self.wait_refill:
                refill_session_id = refill_seq_group.session_id
                if(self.session_dict[refill_session_id].block_status >=BlockStatus.SWAPPING_IN):
                    continue
                else:
                    # need allocate on device memory
                    assert self.session_dict[refill_session_id].block_status ==BlockStatus.ON_HOST
                    this_session_info = self.session_dict[refill_session_id]
                    num_block = this_session_info.seq_group.get_seqs()[0].n_blocks
                    alloc_status = self.block_manager.can_allocate_num_block(num_block)
                    assert alloc_status != AllocStatus.NEVER
                    if(alloc_status == AllocStatus.LATER):
                        break
                    else:
                        assert alloc_status == AllocStatus.OK
                        # we can swap this sequence to gpu
                        seq_group = self.session_dict[refill_session_id].seq_group
                        mapping = self.block_manager.swap_in_refill(seq_group=seq_group)
                        stream_id = self.stream_avail.pop(0)
                        stream_position = len(mapping)
                        # need to extend the kvcacheSwapMeta
                        new_kv_swap_meta.mapping.extend(mapping)
                        new_kv_swap_meta.stream_list.append(stream_id)
                        new_kv_swap_meta.stream_position.append(stream_position)
                        new_kv_swap_meta.sync_this_time = True
                        # print(f"In session {refill_session_id} setting new_kv_swap_meta: mapping:{mapping}")
                        # print(f"In session {refill_session_id} setting new_kv_swap_meta: refill_stream:{stream_id}")
                        #update session_dict
                        this_session_info.current_stream = stream_id
                        this_session_info.block_status = BlockStatus.SWAPPING_IN
                        self.stream_to_session[stream_id] = SetSessionBlockStatus(session_id=refill_session_id,block_status=BlockStatus.ON_DEVICE)
                        # print(f"setting stream {stream_id} maps to action: {refill_session_id} to BlockStatus.ON_DEVICE")
                        # remove from self.refill_requests
                        if(refill_session_id in self.refill_requests):
                            self.refill_requests.remove(refill_session_id)
            if(len(new_kv_swap_meta.stream_list)> 0 ):
                # append this meta to the deque
                self.kv_swap_meta.append(new_kv_swap_meta)
                
            return
            
        if len(self.waiting) + len(self.ready_refill) + len(self.wait_refill) == 0:
            if(self.current_swap_status == SwapStatus.SWAP_OUT):
                # if there are some refill requests waiting
                # issue a sync first
                if len(self.kv_swap_meta)> 0:
                    self.kv_swap_meta[-1].sync_this_time = True
                else:
                    self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                return
                # self.kv_swap_meta.append(KvCacheSwapMeta(SwapStatus.SYNCHRONIZED))
                # return
            # only proceed when self.kv_swap_meta contains only swap_in meta
            not_proceed = any(kv_swap_meta.swap_status == SwapStatus.SWAP_OUT for kv_swap_meta in self.kv_swap_meta)
            if(not_proceed):
                return
            # empty requests pending
            new_kv_swap_meta = KvCacheSwapMeta(SwapStatus.SWAP_IN)
            while len(self.refill_requests) > 0:
                refill_session_id = self.refill_requests[0]
                if(self.session_dict[refill_session_id].block_status >=BlockStatus.SWAPPING_IN):
                    # already load, pop it
                    self.refill_requests.popleft()
                    continue
                # otherwise, try allocate it in gpu
                this_session_info = self.session_dict[refill_session_id]
                num_block = this_session_info.seq_group.get_seqs()[0].n_blocks
                alloc_status = self.block_manager.can_allocate_num_block(num_block)
                assert alloc_status != AllocStatus.NEVER
                if(alloc_status == AllocStatus.LATER):
                    break
                else:
                    assert alloc_status == AllocStatus.OK
                    # we can swap this sequence to gpu
                    seq_group = self.session_dict[refill_session_id].seq_group
                    mapping = self.block_manager.swap_in_refill(seq_group=seq_group)
                    stream_id = self.stream_avail.pop(0)
                    stream_position = len(mapping)
                    # need to extend the kvcacheSwapMeta
                    new_kv_swap_meta.mapping.extend(mapping)
                    new_kv_swap_meta.stream_list.append(stream_id)
                    new_kv_swap_meta.stream_position.append(stream_position)
                    # print(f"In session {refill_session_id} setting new_kv_swap_meta: mapping:{mapping}")
                    # print(f"In session {refill_session_id} setting new_kv_swap_meta: refill_stream:{stream_id}")
                    #update session_dict
                    this_session_info.current_stream = stream_id
                    this_session_info.block_status = BlockStatus.SWAPPING_IN
                    self.stream_to_session[stream_id] = SetSessionBlockStatus(session_id=refill_session_id,block_status=BlockStatus.ON_DEVICE)
                    # print(f"setting stream {stream_id} maps to action: {refill_session_id} to BlockStatus.ON_DEVICE")
            if(len(new_kv_swap_meta.stream_list)> 0 ):
                # append this meta to the deque
                self.kv_swap_meta.append(new_kv_swap_meta)
                
    def _update_wait_refill(self) -> None:
        # iterate through self.wait_refill
        for r_wait_seq_group in list(self.wait_refill):
            session_id = r_wait_seq_group.session_id
            this_session_info = self.session_dict[session_id]
            if(this_session_info.block_status == BlockStatus.ON_DEVICE):
                self.wait_refill.remove(r_wait_seq_group)
                self.ready_refill.append(r_wait_seq_group)
            
    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            running_queue: The queue that contains running requests (i.e.,
                decodes). The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            policy: The sorting policy to sort running_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            A tuple of remaining running queue (should be always 0) after
            scheduling and SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            first_seq = seq_group.get_seqs(SequenceStatus.RUNNING)[0]
            block_len_smaller = len(self.block_manager.block_tables[first_seq.seq_id]) < self.max_model_len/first_seq.block_size
            if(first_seq.truncated and not block_len_smaller):
                # if(first_seq.truncated):
                # need to handle truncated sequence differently
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                continue
            
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        #
                        session_id = victim_seq_group.session_id
                        self.session_dict[session_id].block_status = BlockStatus.NOT_ALLOCATED
                        preempted.append(victim_seq_group)
                    else:
                        # TODO: handle victim seq_group when swap out
                        swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        session_id = seq_group.session_id
                        self.session_dict[session_id].block_status = BlockStatus.NOT_ALLOCATED
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    break
            else:
                session_id = seq_group.session_id
                assert self.session_dict[session_id].block_status == BlockStatus.RUNNING
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            policy: The sorting policy to sort swapped_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining swapped_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        infeasible_seq_groups: List[SequenceGroup] = []

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        
        # prioritized refill ready requests.
        if len(self.ready_refill) > 0:
            refills = self._schedule_refill(budget)
            if(len(refills)> 0):
                seq_groups = refills
        else:
            # schedule regular prefill
            while self._passed_delay(time.time()) and waiting_queue:
                seq_group = waiting_queue[0]

                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_new_tokens = self._get_num_new_tokens(seq_group,
                                                        SequenceStatus.WAITING,
                                                        enable_chunking, budget)
                if not enable_chunking:
                    num_prompt_tokens = waiting_seqs[0].get_len()
                    assert num_new_tokens == num_prompt_tokens

                prompt_limit = self._get_prompt_limit(seq_group)
                if num_new_tokens > prompt_limit:
                    logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds limit of %d", num_new_tokens, prompt_limit)
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    waiting_queue.popleft()
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    # need to check self.refill_info_dict
                    # new_seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
                    # require_block_num = new_seq.n_blocks
                    # self.swap_out_idle_seq(require_block_num)
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds the capacity of block_manager",
                        num_new_tokens)
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    session_id = seq_group.session_id
                    del self.session_dict[session_id]
                    print(f"Deleting session_dict with id {session_id} because prompt len is too long!")
                    waiting_queue.popleft()
                    continue

                lora_int_id = 0
                if self.lora_enabled:
                    lora_int_id = seq_group.lora_int_id
                    assert curr_loras is not None
                    assert self.lora_config is not None
                    if (self.lora_enabled and lora_int_id > 0
                            and lora_int_id not in curr_loras
                            and len(curr_loras) >= self.lora_config.max_loras):
                        # We don't have a space for another LoRA, so
                        # we ignore this request for now.
                        leftover_waiting_sequences.appendleft(seq_group)
                        waiting_queue.popleft()
                        continue

                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break

                # Can schedule this request.
                if curr_loras is not None and lora_int_id > 0:
                    curr_loras.add(lora_int_id)
                waiting_queue.popleft()
                self._allocate_and_set_running(seq_group)
                session_id = seq_group.session_id
                self.session_dict[session_id].block_status = BlockStatus.RUNNING
                self.session_dict[session_id].req_status = RequestStatus.REQ_SCHEDULED
                seq_groups.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                        token_chunk_size=num_new_tokens))
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)

            # Queue requests that couldn't be scheduled.
            waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())
        
        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras, enable_chunking=False)

        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            remaining_running, running_scheduled = self._schedule_running(
                self.running,
                budget,
                curr_loras,
                fcfs_policy,
                enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.swapped, budget, curr_loras, fcfs_policy)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_kick_out=[],
            blocks_to_refill=[],
            kick_out_index=[],
            refill_index=[],
            kick_out_stream=[],
            refill_stream=[],
            stream_to_sync=[],
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self):
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # Decoding should be always scheduled first by fcfs.
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            fcfs_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, fcfs_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_kick_out=[],
            blocks_to_refill=[],
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )
        
    def _schedule_refill(self, budget: SchedulingBudget) -> List[ScheduledSequenceGroup]: 
        refill_seq_groups: List[ScheduledSequenceGroup] = []
        
        while len(self.ready_refill) > 0:
            new_seq_group = self.ready_refill[0]
            session_id = new_seq_group.session_id
            this_session_info = self.session_dict[session_id]
            context_group = this_session_info.seq_group
            assert len(new_seq_group.get_seqs(SequenceStatus.WAITING)) == 1 and len(context_group.get_seqs()) == 1 
            assert this_session_info.block_status==BlockStatus.ON_DEVICE
            new_seq = new_seq_group.get_seqs(SequenceStatus.WAITING)[0]
            context_seq = context_group.get_seqs()[0]
            sampling_params = new_seq_group.sampling_params
            max_num_block = math.ceil(sampling_params.max_tokens/context_seq.block_size)
            total_token_len = len(new_seq.data._prompt_token_ids[1:]) + context_seq.get_len()
            seq_need_block = math.ceil( total_token_len/context_seq.block_size)
            can_allocate = self.block_manager.append_slots_refill(context_seq,max_num_block,seq_need_block)
            if(can_allocate==AllocStatus.LATER):
                break
            if(can_allocate==AllocStatus.NEVER):
                logger.warning(
                "Input prompt (%d tokens) is too long"
                " and exceeds the capacity of block_manager",
                new_seq_group.get_num_uncomputed_tokens())
            # otherwise we can allocate the new sequence with context
            # need to update the new sequence group to include its context
            self.ready_refill.popleft()
            context_group.set_seq_group_status()
            context_group.sampling_params = new_seq_group.sampling_params
            num_new_seqs = new_seq_group.get_max_num_running_seqs()
            context_seq.inputs['prompt_token_ids'].extend(context_seq.data._output_token_ids[:-1])
            context_seq.inputs['prompt_token_ids'].extend(new_seq.data._prompt_token_ids[1:])
            context_seq.status = SequenceStatus.RUNNING
            context_seq.stop_reason = None
            context_seq.output_text = ""
            old_data  = context_seq.data
            context_seq.data = SequenceData(context_seq.inputs["prompt_token_ids"])
            context_seq.data.truncated_len = old_data.truncated_len
            context_seq.data._num_computed_tokens = old_data._num_computed_tokens
            num_new_tokens = context_seq.get_num_new_tokens()
            context_seq.tokens.pop()
            context_seq.tokens.extend(new_seq.inputs['prompt'])
            context_seq.read_offset = context_seq.prefix_offset+1
            context_group.request_id = new_seq_group.request_id
            context_group.is_last_round = new_seq_group.is_last_round
            context_group.metrics = new_seq_group.metrics
            budget.add_num_batched_tokens(new_seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(new_seq_group.request_id, num_new_seqs)
            this_session_info.block_status = BlockStatus.RUNNING
            this_session_info.req_status = RequestStatus.REQ_SCHEDULED
            if(context_seq.data.get_len()>=sampling_params.max_tokens):
                # print("hitting here \nhitting here \nhitting here \n")
                cur_len = context_seq.data.get_len()
                num_blocks_truncated = math.ceil((cur_len - sampling_params.max_tokens)/context_seq.block_size)
                if(cur_len - sampling_params.max_tokens) % context_seq.block_size == 0:
                    num_blocks_truncated +=1
                self.block_manager.truncate_blocks(seq=context_seq,num=num_blocks_truncated)
                num_tokens_truncated = num_blocks_truncated * context_seq.block_size
                context_seq.data.truncated_len += num_tokens_truncated
                context_seq.truncated = True
                context_seq.read_offset -= num_tokens_truncated
                context_seq.prefix_offset -= num_tokens_truncated
                context_seq.data._num_computed_tokens -= num_tokens_truncated
                # print(f"current length: {cur_len}")
                # print(f"num block truncated: {num_blocks_truncated}")
                # print(f"sequence truncated len: {context_seq.data.truncated_len}")
                # print(f"\n\nnum tokens computed in refill: {context_seq.data._num_computed_tokens}\n\n")
            refill_seq_groups.append(ScheduledSequenceGroup(seq_group=context_group,token_chunk_size=num_new_tokens))
            this_session_info.seq_group = context_group
        return refill_seq_groups

    def process_truncated_seqs(self)-> None:
        for seq_group in self.running:
            truncated_seqs = seq_group.get_seqs(SequenceStatus.TRUNCATED)
            if(truncated_seqs is None or len(truncated_seqs)==0):
                continue
            else:
                for truncated_seq in truncated_seqs:
                    self.block_manager.truncate_first_append_last(truncated_seq)
                    truncated_seq.status = SequenceStatus.RUNNING
                    truncated_seq.truncated = True
                    truncated_seq.data._num_computed_tokens -= truncated_seq.block_size
                    truncated_seq.read_offset -= truncated_seq.block_size
                    truncated_seq.prefix_offset -= truncated_seq.block_size
                    
    def process_kv_swap_meta(self, scheduler_outputs: SchedulerOutputs)-> None:
        if len(self.kv_swap_meta) == 0:
            # nothing to process
            return
        print(f"length of kv_swap_meta:{len(self.kv_swap_meta)}")
        if(len(self.temp_finished)> 0):
            print(f"length of temp finished:{len(self.temp_finished)}")
        new_kv_meta = self.kv_swap_meta[0]
        if(new_kv_meta.swap_status == SwapStatus.SYNCHRONIZED):
            # TODO call synchronize
            self.synchronize_stream(scheduler_outputs)
            # print(f"len(self.kv_swap_meta): {len(self.kv_swap_meta)}")
            self.kv_swap_meta.popleft()
            return
        
        if(new_kv_meta.swap_status == SwapStatus.SWAP_OUT):
            if(self.current_swap_status == SwapStatus.SWAP_IN):
                # TODO call synchronize
                self.synchronize_stream(scheduler_outputs)
                # print(f"len(self.kv_swap_meta): {len(self.kv_swap_meta)}")
                return
            else:
                while(len(self.kv_swap_meta)> 0 and self.kv_swap_meta[0].swap_status == SwapStatus.SWAP_OUT):
                    # we can assume current_swap_status == SWAP_OUT or SYNCHRONIZED
                    new_kv_meta = self.kv_swap_meta[0]
                    scheduler_outputs.kick_out_index.extend(new_kv_meta.stream_position)
                    scheduler_outputs.blocks_to_kick_out.extend(new_kv_meta.mapping)
                    scheduler_outputs.kick_out_stream.extend(new_kv_meta.stream_list)
                    self.current_swap_status = SwapStatus.SWAP_OUT
                    self.stream_pend_sync.extend(new_kv_meta.stream_list)
                    if(new_kv_meta.sync_this_time):
                        self.kv_swap_meta.popleft()
                        self.synchronize_stream(scheduler_outputs)
                        return
                        # print(f"len(self.kv_swap_meta): {len(self.kv_swap_meta)}")
                    self.kv_swap_meta.popleft()
                return
            
        if(new_kv_meta.swap_status == SwapStatus.SWAP_IN):
            if(self.current_swap_status == SwapStatus.SWAP_OUT):
                # TODO call synchronize
                self.synchronize_stream(scheduler_outputs)
                # print(f"len(self.kv_swap_meta): {len(self.kv_swap_meta)}")
                return
            else:
                
                scheduler_outputs.refill_index = new_kv_meta.stream_position
                scheduler_outputs.blocks_to_refill = new_kv_meta.mapping
                scheduler_outputs.refill_stream = new_kv_meta.stream_list
                self.current_swap_status = SwapStatus.SWAP_IN
                # print(f"setting scheduler outputs: refill index:{scheduler_outputs.refill_index}")
                # print(f"setting scheduler outputs: refill_stream:{scheduler_outputs.refill_stream}")
                self.stream_pend_sync.extend(new_kv_meta.stream_list)
                if(new_kv_meta.sync_this_time):
                    self.synchronize_stream(scheduler_outputs)
                    # print(f"len(self.kv_swap_meta): {len(self.kv_swap_meta)}")
                self.kv_swap_meta.popleft()
                return
            
            
    def synchronize_stream(self, scheduler_outputs: SchedulerOutputs)-> None:
        scheduler_outputs.stream_to_sync = [1]
        if(self.current_swap_status == SwapStatus.SWAP_IN or self.current_swap_status == SwapStatus.SYNCHRONIZED):
            self.block_manager.sync_free_block_table('cpu')
        if(self.current_swap_status == SwapStatus.SWAP_OUT or self.current_swap_status == SwapStatus.SYNCHRONIZED):
            self.block_manager.sync_free_block_table('gpu')
        
        for stream_id in self.stream_pend_sync:
            set_action = self.stream_to_session[stream_id]
            if(set_action != None and set_action.block_status != None):
                self.session_dict[set_action.session_id].block_status = set_action.block_status
                # print(f"Setting {set_action.session_id} block status to {set_action.block_status}")
                # print(f"scheduler_outputs.refill_stream {scheduler_outputs.refill_stream}")
                # print(f"scheduler_outputs.refill_index {scheduler_outputs.refill_index}")
            del self.stream_to_session[stream_id]
        # print(f"syncing following stream: {self.stream_pend_sync}")    
        self.stream_avail.extend(self.stream_pend_sync)
        self.stream_pend_sync = []
        self.current_swap_status = SwapStatus.SYNCHRONIZED
            

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        # cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        # assert cuda_visible_device == "0"
        schedule_begin = time.time()
        # need to handle truncated sequences first
        if(self.use_truncation):
            self.process_truncated_seqs()
        # scheduler_outputs = None
        # if(len(self.refill_wait)>0 and len(self.context_req_id_ref)>0) and self._passed_delay_refill(time.time()):
        #     # print("entering refill schedulen\n\n\n\n\n\n\nentering refill schedule\n\n")
        #     scheduler_outputs = self.refill_schedule()
        #     if(len(scheduler_outputs.scheduled_seq_groups)== 0):
        #         scheduler_outputs = self._schedule()
        # else: 
        scheduler_outputs = self._schedule()
        now = time.time()
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)
                # if(self.refilling_mode):
                    # print("output text: ",seq.output_text)
                    # print("output prob log: ",seq.output_logprobs)
                    # print("computed_length: ",seq.data.get_num_computed_tokens())
                    # print("uncomputed: ", seq.data.get_num_uncomputed_tokens())
                    # print("seq length: ",seq.data.get_len())
                    # print()

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))
            # if(self.refilling_mode and need_common_computed):
            #     active_seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            #     num_computed_tokens=active_seq.data.get_num_computed_tokens()
            #     common_computed_block_nums = (self.block_manager.get_block_table(active_seq))[:(num_computed_tokens//active_seq.block_size)]
            #     # print("common_computed_block_nums: ",common_computed_block_nums)

            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens() <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
                prompt_adapter_request=seq_group.prompt_adapter_request,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)
        self.try_store_kv_cache()
        if(not self.pause_load_kv):
            self.try_load_kv_cache()
        self.process_kv_swap_meta(scheduler_outputs=scheduler_outputs)
        self._update_wait_refill()
        if(len(self.kv_swap_meta)> self.trigger_sync_threshold):
            print(f"self.kv_swap_meta: {self.kv_swap_meta}")
        # print(f"kv_swap_meta at 1636:{self.kv_swap_meta}")
        self.time_in_scheduler +=time.time()-schedule_begin
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        # self.block_manager.free(seq)
        self.wait_for_free.append(seq)

    def free_finished_seq_groups(self) -> None:
        for queue in [self.running, self.swapped, self.waiting]:
            self._finished_requests_ids += [
                seq_group.request_id for seq_group in queue
                if seq_group.is_finished()
            ]
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())
        
    def swap_finished_seq_groups(self) -> None:
        for queue in [self.running, self.swapped, self.waiting]:
            self._finished_requests_ids += [
                seq_group.request_id for seq_group in queue
                if seq_group.is_finished()
            ]
        need_to_free = [seq_group for queue in [self.running, self.swapped, self.waiting] for seq_group in queue
                             if (seq_group.is_finished() and seq_group.is_last_round)]
        new_finished = [seq_group for queue in [self.running, self.swapped, self.waiting] for seq_group in queue
                             if (seq_group.is_finished() and not seq_group.is_last_round)]
        for f_seq_group in new_finished:
            self.session_dict[f_seq_group.session_id].req_status = RequestStatus.TEMP_FINISHED
        self.temp_finished.extend(new_finished)
        if(len(self.temp_finished)> 0):
            # print(f"Before: self.cureent swap status:{self.current_swap_status}")
            # print(f"Before: current kv meta : {self.kv_swap_meta}")
            self.try_store_kv_cache()
            # print(f"After: self.cureent swap status:{self.current_swap_status}")
            # print(f"After: current kv meta : {self.kv_swap_meta}")
            # print(f"After: self.temp_finished:{self.temp_finished}")
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())
        temp_finished_seq_id = [seq_id for seq_group in new_finished for seq_id in seq_group.seqs_dict.keys() ]
        # for seq_group in self.hung:
        #     for seq_id in seq_group.seqs_dict.keys():
        #         temp_finished_seq_id.append(seq_id) 
        
        self.wait_for_free = [seq for seq in self.wait_for_free if seq.seq_id not in temp_finished_seq_id] 
        for seq in self.wait_for_free:
            self.block_manager.free(seq)
            
        for instance_seq_group in need_to_free:
            for seq in instance_seq_group.get_seqs():
                print(f"Free seq {seq.seq_id} !\n\n")
                self.block_manager.free(seq)
            session_id = instance_seq_group.session_id
            del self.session_dict[session_id]
            print(f"Deleting session_dict with id {session_id} because it is already last round!")
            
        self.wait_for_free = []
        

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 5 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
            logger.warning(f"seq_group has len: {seq_group.get_seqs()[0].get_len()}")
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay
    
    def _passed_delay_refill(self, now: float) -> bool:
        if self.prev_prompt_refill:
            self.last_prompt_latency_refill = now - self.prev_time_refill
        self.prev_time_refill, self.prev_prompt_refill = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.refill_wait:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.refill_wait])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency_refill)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns 0 if the new token cannot be computed due to token budget.
        """
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        if(num_new_tokens <= 0):
            print(f"seq_group: {seq_group}")
            print(f"seqs: {seq_group.get_seqs()}")
            print("seq_len with num_new_tokens < 0",seqs[0].get_len())
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens
