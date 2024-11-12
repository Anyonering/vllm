import asyncio
import logging

import grpc
import chat_pb2
import chat_pb2_grpc
import pdb
import signal
import os

import vllm
from vllm.entrypoints.llm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs

import asyncio
import time
import uuid
from typing import Optional

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


MODEL_PATH = '~/personal/projects/vllm_inference/model_data/opt-1.3b/'

class LlmEngine(chat_pb2_grpc.LlmEngineServicer):
    def __init__(self, *args, **kwargs):
        model_dir = os.path.expanduser(MODEL_PATH)
        self.engine =  vllm.AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_dir,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8))
        self.output_len = []
        self.ttft_list = []
        self.run_time = []
        self.start_time = 0
        
    
    async def processChatReq(self, request: chat_pb2.ChatReq, context: grpc.aio.ServicerContext):
        print(f"receive Request with session id {request.session_id}, and request id {request.request_id} with prompt length {len(request.prompt)}")
        self.engine.engine.scheduler[0].is_finish_dict[request.session_id] = request.is_last
        if(self.start_time == 0):
            self.start_time = time.time()
        results_generator = self.engine.generate(
        request.prompt,
        vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, min_tokens=20,),
        request_id=request.request_id,
        session_id = request.session_id,
        # refill_requests=refill_requests
        )
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        self.output_len.append(len(final_output.outputs[0].token_ids))
        self.ttft_list.append(final_output.metrics.first_token_time - final_output.metrics.arrival_time)
        self.run_time.append(final_output.metrics.last_token_time - final_output.metrics.first_scheduled_time)
        # prompt = final_output.prompt
        text_output = [output.text for output in final_output.outputs]
        return chat_pb2.ChatResp(answer=text_output[0])
    
    async def processInfoReq(self, request: chat_pb2.InfoReq, context: grpc.aio.ServicerContext):
        self.engine.engine.scheduler[0].refill_requests.append(request.session_id)
        return chat_pb2.InfoResp(success=True)
        
    
async def serve() -> None:
    server = grpc.aio.server()
    engine = LlmEngine()
    chat_pb2_grpc.add_LlmEngineServicer_to_server(engine, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    
    await server.start()
    termination_event = asyncio.Event()

    # Signal handler for graceful shutdown
    def handle_sigint():
        print("\nReceived termination signal (Ctrl+C)")
        # print(f"output len list: {engine.output_len}")
        termination_event.set()  # Set the event to start the shutdown process

    # Register SIGINT (Ctrl+C) signal handler
    # signal.signal(signal.SIGINT, lambda s, f: handle_sigint())
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handle_sigint)
    try:
        # Wait until termination signal is received
        await termination_event.wait()
    except KeyboardInterrupt:
        server.stop(0)
        pass
    finally:
        # Print the output length list before shutting down
        print(f"output len list: {engine.output_len}")
        print(f"time to first token list: {engine.ttft_list}")
        print(f"Run time list: {engine.run_time}")
        print(f"Last for {time.time()- engine.start_time}")
        await server.stop(0)  # Stop the server immediately
    
    
    # try:
    #     await server.start()
    #     await server.wait_for_termination()
    # except KeyboardInterrupt:
    #     print(f"output len list: {engine.output_len}")
    #     server.stop(0)
        
    # await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(serve())