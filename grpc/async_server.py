import asyncio
import logging

import grpc
import chat_pb2
import chat_pb2_grpc

import os

import vllm
from vllm.entrypoints.llm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs

import asyncio
import time
import uuid
from typing import Optional


MODEL_PATH = '~/personal/projects/vllm_inference/model_data/opt-1.3b/'

class LlmEngine(chat_pb2_grpc.LlmEngineServicer):
    def __init__(self, *args, **kwargs):
        model_dir = os.path.expanduser(MODEL_PATH)
        self.engine =  vllm.AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_dir,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=256,
        )
    )
    async def processChatReq(self, request: chat_pb2.ChatReq, context: grpc.aio.ServicerContext):
        results_generator = self.engine.generate(
        request.prompt,
        vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128, min_tokens=20,),
        request_id=request.request_id,
        session_id = request.session_id,
        # refill_requests=refill_requests
        )
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # prompt = final_output.prompt
        text_output = [output.text for output in final_output.outputs]
        return chat_pb2.ChatResp(answer=text_output[0])
    
    async def processInfoReq(self, request: chat_pb2.InfoReq, context: grpc.aio.ServicerContext):
        self.engine.engine.scheduler[0].refill_requests.append(request.session_id)
        return chat_pb2.InfoResp(success=True)
        
    
async def serve() -> None:
    server = grpc.aio.server()
    chat_pb2_grpc.add_LlmEngineServicer_to_server(LlmEngine(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())