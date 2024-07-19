import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from .scheduler import Scheduler
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

class Session:
    # Session is a list of chat between user and the model.
    # It will use the chat history as the context for later responses.
    seq_group: SequenceGroup
    
    # the length of tokens with kv cache computed
    context_len: int
    
    # the maximum  number of tokens in total for a session
    max_token_len: int
    
    # the request_id of the session
    req_id: str
    
    # the length of the tokens whose kv cache have not been computed (or lost 
    # because they are at the last page)
    new_prompt_seq_len: int
    
    