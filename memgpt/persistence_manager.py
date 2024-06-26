from typing import List

from memgpt.memory import (
    RecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.data_types import Message, AgentState


class PersistenceManager:

    def __init__(self, agent_state: AgentState):
        self.archival_memory = EmbeddingArchivalMemory(agent_state)
        self.recall_memory = RecallMemory(agent_state)

    def persist_messages(self, added_messages: List[Message]):
        self.recall_memory.insert_many([m for m in added_messages])
