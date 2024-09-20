
import asyncio
import time
import logging

import grpc
import chat_pb2
import chat_pb2_grpc


task_list = []
def blocking_task():
    print("doing other work")
    time.sleep(3)
    print("sleeping call ----- after")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chat_pb2_grpc.LlmEngineStub(channel)
        request = chat_pb2.ChatReq(prompt="Who is the most powerful person in the world?",request_id="4",session_id=4)
        task = asyncio.create_task(make_grpc_request(stub, request))
        print("sleeping call ----- before")
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        coro = asyncio.to_thread(blocking_task)
        await coro
        task_list.append(task)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("run function time: ",current_time)
        # await task_list[0]
        # response = stub.processChatReq(chat_pb2.ChatReq(prompt="Who is the most powerful person in the world?",request_id="1",session_id=1))
        # stub2 = chat_pb2_grpc.LlmEngineStub(channel)
        # resp2 = await stub.processInfoReq(chat_pb2.InfoReq(session_id=0))
        # task_list.append(response)
        # real_resp = await response
        # print("Greeter client received: " + real_resp.answer)
        # with grpc.insecure_channel("localhost:50051") as channel:
        #     stub = chat_pb2_grpc.LlmEngineStub(channel)
        #     stub.processInfoReq(chat_pb2.InfoReq(session_id=0))
        # print("Greeter client received: " + response.answer)
        # print(resp2)
    
async def make_grpc_request(stub, request):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("time making request: ",current_time)
    response = await stub.processChatReq(request)
    print("Response received:", response.answer)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("time finished request: ",current_time)
if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(run())
    