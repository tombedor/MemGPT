from typing import Optional, List

from memgpt.agent_store.storage import StorageConnector
from memgpt.config import MemGPTConfig
from memgpt.utils import get_local_time
from memgpt.data_types import Message, Passage, AgentState
from memgpt.embeddings import query_embedding, parse_and_chunk_text


class RecallMemory:
    """Recall memory based on base functions implemented by storage connectors"""

    def __init__(self, agent_state):
        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)

        self.agent_state = agent_state

        # create storage backend
        self.storage = StorageConnector.get_recall_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def text_search(self, query_string, count=None, start=None):  # type: ignore
        results = self.storage.query_text(query_string, count, start)  # type: ignore
        results_json = [message.to_openai_dict() for message in results]  # type: ignore
        return results_json, len(results)

    def date_search(self, start_date, end_date, count=None, start=None):
        results = self.storage.query_date(start_date, end_date, count, start)  # type: ignore
        results_json = [message.to_openai_dict() for message in results]  # type: ignore
        return results_json, len(results)

    def insert(self, message: Message):
        self.storage.insert(message)

    def insert_many(self, messages: List[Message]):
        self.storage.insert_many(messages)

    def __len__(self):
        return self.storage.size()


class EmbeddingArchivalMemory:
    """Archival memory with embedding based search"""

    def __init__(self, agent_state: AgentState, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archival_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """
        self.storage = StorageConnector.get_archival_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)

        self.top_k = top_k
        self.agent_state = agent_state

        # create embedding model
        # create storage backend
        self.cache = {}

    def create_passage(self, text, embedding):
        return Passage(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            text=text,
            embedding=embedding,
            embedding_dim=MemGPTConfig.embedding_dim,
            embedding_model=MemGPTConfig.embedding_model_name,
        )

    def insert(self, memory_string):
        """Embed and save memory string"""

        if not isinstance(memory_string, str):
            return TypeError("memory must be a string")

        try:
            passages = []
            embedding_model = MemGPTConfig.embedding_model

            # breakup string into passages
            for text in parse_and_chunk_text(memory_string):  # type: ignore
                embedding = embedding_model.get_text_embedding(text)
                # fixing weird bug where type returned isn't a list, but instead is an object
                # eg: embedding={'object': 'list', 'data': [{'object': 'embedding', 'embedding': [-0.0071973633, -0.07893023,
                if isinstance(embedding, dict):
                    try:
                        embedding = embedding["data"][0]["embedding"]  # type: ignore
                    except (KeyError, IndexError):
                        # TODO as a fallback, see if we can find any lists in the payload
                        raise TypeError(
                            f"Got back an unexpected payload from text embedding function, type={type(embedding)}, value={embedding}"
                        )
                passages.append(self.create_passage(text, embedding))

            # insert passages
            self.storage.insert_many(passages)
            return True
        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query_string, count=None, start=None):  # type: ignore
        """Search query string"""
        if not isinstance(query_string, str):
            return TypeError("query must be a string")

        try:
            if query_string not in self.cache:
                # self.cache[query_string] = self.retriever.retrieve(query_string)
                query_vec = query_embedding(query_string)
                self.cache[query_string] = self.storage.query(query_string, query_vec, top_k=self.top_k)  # type: ignore

            start = int(start if start else 0)
            count = int(count if count else self.top_k)  # type: ignore
            end = min(count + start, len(self.cache[query_string]))

            results = self.cache[query_string][start:end]
            results = [{"timestamp": get_local_time(), "content": node.text} for node in results]
            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def __len__(self):
        return self.storage.size()
