import time
import json
import numpy as np
import random
import pdb

import asyncio
import grpc
import chat_pb2
import chat_pb2_grpc
import threading
from transformers import PreTrainedTokenizerBase, AutoTokenizer


class ChatDataLoader(object):
    def __init__(
        self,
        # assuming normal distribution
        mean_concurrent_users,  # number of concurrent users
        deviation_concurrent_users,
        mean_read_rate,  # rate for reading
        deviation_read_rate,
        mean_type_rate,  # rate of typing
        deviation_type_rate,
    ):

        self.mean_concurrent_users = mean_concurrent_users
        self.deviation_concurrent_users = deviation_concurrent_users
        self.mean_read_rate = mean_read_rate
        self.deviation_read_rate = deviation_read_rate
        self.mean_type_rate = mean_type_rate
        self.deviation_type_rate = deviation_type_rate
        self.continue_task = 0.5

        # distribution shift
        self.normal_distribution = np.random.default_rng()
        # this is potentially wrong need to switch to random int within a min and max
        # self.num_current_clients = abs(
        #     int(
        #         self.normal_distribution.normal(
        #             self.mean_concurrent_users, self.deviation_concurrent_users
        #         )
        #     )
        # )
        # simplify for checking correctness and debugging
        self.num_current_clients = 30
        # numbers of word read
        print(f"Num current clients {self.num_current_clients}")
        self.crps = abs(
            self.normal_distribution.normal(
                self.mean_read_rate,
                self.deviation_read_rate,
                size=self.num_current_clients,
            )
        )
        # number of words typed
        self.wsps = abs(
            self.normal_distribution.normal(
                self.mean_type_rate,
                self.deviation_type_rate,
                size=self.num_current_clients,
            )
        )
        # PATH = "/users/TA744/sharegpt90k/sg_90k_part1.json"
        PATH = "/home/anyong/Downloads/sg_90k_part1.json"
        MODEL_PATH = '/home/anyong/personal/projects/vllm_inference/model_data/opt-1.3b/'
        with open(PATH, "r") as fopen:
            self.open_data = json.load(fopen)
        # a list which contains active connections
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.max_prompt_len = 2048
        self.active_sessions = dict()
        self.next_req_data = dict()
        self.next_req_time = dict()
        self.next_info_req_time = dict()
        self.client_id = 0
        self.task_list = []
        self.request_id_counter = 0
        self.counter_lock = threading.Lock()
        self.dict_lock = threading.Lock()
        self.tokenizer_lock = threading.Lock()
        self.session_counter = 0
        self.interval_between_sessions = 3
        return None

    async def rpc_call(self, req_data, client_id, is_last, token_len):
        """
        RPC calls for sending requests to the server
        """
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            
            stub = chat_pb2_grpc.LlmEngineStub(channel)
            input = req_data
            # print("This is the input sent: ",input)
            new_req_id = None
            with self.counter_lock:
                self.request_id_counter += 1
                new_req_id = str(self.request_id_counter)
            print(f"Ready to send request with id {new_req_id}, with input length {token_len} and content: {input[:100]}!")
            
            request = chat_pb2.ChatReq(
                prompt=input, request_id=new_req_id
            )
            # pdb.set_trace()
            response = await stub.processChatReq(request)
            answer_len = len(response.answer)
            print( "Receive llm text: ",response.answer[:50])
            return answer_len, response.answer
            # pdb.set_trace()
            
    def thread_rpc_call(self, loop, req_data, client_id):
        # report a message
        # print('thread running')
        # wait a moment
        # create a coroutine
        coro = self.rpc_call(req_data, client_id)
        # execute a coroutine
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # wait for the task to finish
        future.result()
        # report a message
        # print('thread done')
        
    def exceed_length_limit(self, req_data):
        if(req_data['from']=='human' and len(req_data['value'])> 20000):
            return True
        return False
        
        
    async def sleep_time(self, sleep_time):
        await asyncio.sleep(sleep_time)
        
    def run_thread_loop(self, loop, rank_client):
        # try:
        #     result = await asyncio.wait_for(long_running_task(), timeout=5.0)
        #     print("Task completed successfully:", result)
        # except asyncio.TimeoutError:
        #     print("Task timed out after 5 seconds.")
        input_data = None
        interval_chat_req = 0
        interval_info_req = 0
        cur_client_id = rank_client
        is_last = True
        with self.dict_lock:
            while(True):
                input_data = self.open_data.pop(0)["conversations"]
                result = any(self.exceed_length_limit(chat_data) for chat_data in input_data)
                if(result):
                    continue
                if(input_data[0]['from']=="human"):
                    break
            cur_client_id = self.session_counter
            self.session_counter +=1
            
        # with self.dict_lock:
        #     input_data = self.open_data.pop(0)["conversations"]
        #     cur_client_id = self.session_counter
        #     self.session_counter +=1
        req_data = input_data.pop(0)
        assert req_data["from"] == "human"
        total_text = req_data["value"]
        
        while True:
            is_last = True
            
            coro = self.sleep_time(interval_chat_req)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()
            # req_data = input_data.pop(0)
            # assert req_data["from"] == "human"
            if len(input_data)>1:
                # there are at least two more
                is_last = False
                next_recv = input_data.pop(0)
                assert next_recv["from"] == "gpt"
                next_send = input_data.pop(0)
                assert next_send["from"] == "human"
                interval_chat_req = len(next_send["value"])/self.mean_type_rate
            len_token = 0
            with self.tokenizer_lock:
                len_token = len(self.tokenizer.encode(total_text)[-(self.max_prompt_len-8):])
                print(f"")
                text_send = self.tokenizer.decode(self.tokenizer.encode(total_text)[-(self.max_prompt_len-8):])
            coro = self.rpc_call(text_send, cur_client_id, is_last=is_last,token_len=len_token)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            answer_len, answer_text = future.result()
            total_text += answer_text
            # if(not is_last)
            #     req_data = next_send
            # TODO Need to be changed to match the actually returned token num
            # interval_info_req = min(answer_len/self.mean_read_rate, 20)
            interval_info_req = 5
            # try:
                
            #     next_recv = self.active_sessions[client_id].pop(0)
            #     assert next_recv["from"] == "gpt"
            #     next_send = self.active_sessions[client_id].pop(0)
            #     assert next_send["from"] == "human"
            # except Exception as e:
            #     pass
            # if len(input_data)> 1:
                
                
            
            if(not is_last):
                req_data = next_send
                total_text += req_data['value']
                coro = self.sleep_time(interval_info_req)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                future.result()
                
            else:
                # need to handle processing a new list from dict
                with self.dict_lock:
                    while(True):
                        input_data = self.open_data.pop(0)["conversations"]
                        result = any(self.exceed_length_limit(chat_data) for chat_data in input_data)
                        if(result):
                            continue
                        if(input_data[0]['from']=="human"):
                            break
                    cur_client_id = self.session_counter
                    self.session_counter +=1
                req_data = input_data.pop(0)
                # if(req_data["from"]!= "human"):
                #     print(f"1: {req_data['from']}")
                #     for i in range(len(input_data)):
                #         print(f"{i+2}: {input_data[i]['from']}")
                assert req_data["from"] == "human"
                total_text = req_data["value"]
                coro = self.sleep_time(self.interval_between_sessions)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                future.result()
            

    async def send_data(self):
        """
        Based on number of users and prompt size decide.
        """
        loop = asyncio.get_running_loop()
        for single_client in range(self.num_current_clients):
            threading.Thread(target=self.run_thread_loop, args=(loop,single_client)).start()

        # get the conversations in the list for client id

        await asyncio.sleep(300)

if __name__ == "__main__":
    dataloader = ChatDataLoader(10, 5, 25, 5, 20, 5)
    asyncio.run(dataloader.send_data())
