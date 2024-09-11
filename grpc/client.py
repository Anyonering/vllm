
import asyncio
import logging

import grpc
import chat_pb2
import chat_pb2_grpc

async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chat_pb2_grpc.LlmEngineStub(channel)
        response = await stub.processChatReq(chat_pb2.ChatReq(prompt="Who is the most powerful person in the world?",request_id="1",session_id=1))
        # stub2 = chat_pb2_grpc.LlmEngineStub(channel)
        # resp2 = await stub.processInfoReq(chat_pb2.InfoReq(session_id=0))
        
    # with grpc.insecure_channel("localhost:50051") as channel:
    #     stub = chat_pb2_grpc.LlmEngineStub(channel)
    #     stub.processInfoReq(chat_pb2.InfoReq(session_id=0))
    print("Greeter client received: " + response.answer)
    # print(resp2)
if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(run())