from typing import Optional, List

from memgpt.agent_store.storage import StorageConnector
from memgpt.constants import MESSAGE_SUMMARY_WARNING_FRAC
from memgpt.utils import get_local_time, printd, count_tokens
from memgpt.prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from memgpt.llm_api_tools import create
from memgpt.data_types import Message, Passage, AgentState
from memgpt.embeddings import embedding_model, query_embedding, parse_and_chunk_text


class InContextMemory(object):
    """Held in-context inside the system message

    Core Memory: Refers to the system block, which provides essential, foundational context to the AI.
    This includes the persona information, essential user details,
    and any other baseline data you deem necessary for the AI's basic functioning.
    """

    def __init__(self, persona=None, human=None, persona_char_limit=None, human_char_limit=None, archival_memory_exists=True):
        self.persona = persona
        self.human = human
        self.persona_char_limit = persona_char_limit
        self.human_char_limit = human_char_limit

        # affects the error message the AI will see on overflow inserts
        self.archival_memory_exists = archival_memory_exists

    def __repr__(self) -> str:
        return f"\n### CORE MEMORY ###" + f"\n=== Persona ===\n{self.persona}" + f"\n\n=== Human ===\n{self.human}"

    def to_dict(self):
        return {
            "persona": self.persona,
            "human": self.human,
        }

    @classmethod
    def load(cls, state):
        return cls(state["persona"], state["human"])

    def edit_persona(self, new_persona):
        if self.persona_char_limit and len(new_persona) > self.persona_char_limit:
            error_msg = f"Edit failed: Exceeds {self.persona_char_limit} character limit (requested {len(new_persona)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'persona' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.persona = new_persona
        return len(self.persona)

    def edit_human(self, new_human):
        if self.human_char_limit and len(new_human) > self.human_char_limit:
            error_msg = f"Edit failed: Exceeds {self.human_char_limit} character limit (requested {len(new_human)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'human' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.human = new_human
        return len(self.human)


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize,
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_state.llm_config.context_window

    assert context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = str(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
        trunc_ratio = (MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [summarize_messages(agent_state, message_sequence_to_summarize=message_sequence_to_summarize[:cutoff])]
            + message_sequence_to_summarize[cutoff:]
        )
    message_sequence = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": summary_input},
    ]

    response = create(
        agent_state=agent_state,
        messages=message_sequence,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply


class RecallMemory:
    """Recall memory based on base functions implemented by storage connectors"""

    def __init__(self, agent_state):
        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)

        self.agent_state = agent_state

        # create embedding model
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size

        # create storage backend
        self.storage = StorageConnector.get_recall_storage_connector(user_id=agent_state.user_id, agent_id=agent_state.id)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def text_search(self, query_string, count=None, start=None):  # type: ignore
        results = self.storage.query_text(query_string, count, start)
        results_json = [message.to_openai_dict() for message in results]
        return results_json, len(results)

    def date_search(self, start_date, end_date, count=None, start=None):
        results = self.storage.query_date(start_date, end_date, count, start)
        results_json = [message.to_openai_dict() for message in results]
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
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size
        assert self.embedding_chunk_size, f"Must set {agent_state.embedding_config.embedding_chunk_size}"

        # create storage backend
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def create_passage(self, text, embedding):
        return Passage(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            text=text,
            embedding=embedding,
            embedding_dim=self.agent_state.embedding_config.embedding_dim,
            embedding_model=self.agent_state.embedding_config.embedding_model,
        )

    def insert(self, memory_string):
        """Embed and save memory string"""

        if not isinstance(memory_string, str):
            return TypeError("memory must be a string")

        try:
            passages = []

            # breakup string into passages
            for text in parse_and_chunk_text(memory_string, self.embedding_chunk_size):  # type: ignore
                embedding = self.embed_model.get_text_embedding(text)
                # fixing weird bug where type returned isn't a list, but instead is an object
                # eg: embedding={'object': 'list', 'data': [{'object': 'embedding', 'embedding': [-0.0071973633, -0.07893023,
                if isinstance(embedding, dict):
                    try:
                        embedding = embedding["data"][0]["embedding"]
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
                query_vec = query_embedding(self.embed_model, query_string)
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
