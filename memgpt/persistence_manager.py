from typing import List

from memgpt.memory import (
    RecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.utils import printd
from memgpt.data_types import Message, AgentState


class PersistenceManager:

    def __init__(self, agent_state: AgentState):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        self.archival_memory = EmbeddingArchivalMemory(agent_state)
        self.recall_memory = RecallMemory(agent_state)

    def prepend_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.prepend_to_message")
        # self.messages = [self.messages[0]] + added_messages + self.messages[1:]

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])

    def append_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.append_to_messages")
        # self.messages = self.messages + added_messages

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])
