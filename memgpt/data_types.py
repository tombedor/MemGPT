""" This module contains the data types used by MemGPT. Each data type must include a function to create a DB model. """

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, TypeVar
import numpy as np

from memgpt.constants import DEFAULT_HUMAN_TEXT, MAX_EMBEDDING_DIM, NON_USER_MSG_PREFIX
from memgpt.utils import create_uuid_from_string

from pydantic import BaseModel, Field

from pydantic import BaseModel, Field


class Record:
    """
    Base class for an agent's memory unit. Each memory unit is represented in the database as a single row.
    Memory units are searched over by functions defined in the memory classes
    """

    def __init__(self, id: Optional[uuid.UUID] = None):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"


# This allows type checking to work when you pass a Passage into a function expecting List[Record]
# (just use List[RecordType] instead)
RecordType = TypeVar("RecordType", bound="Record")


class ToolCall(object):
    def __init__(
        self,
        id: str,
        tool_call_type: str,  # only 'function' is supported
        # function: { 'name': ..., 'arguments': ...}
        function: Dict[str, str],
    ):
        self.id = id
        self.tool_call_type = tool_call_type
        self.function = function

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.tool_call_type,
            "function": self.function,
        }


class Message(Record):
    """Representation of a message sent.

    Messages can be:
    - agent->user (role=='agent')
    - user->agent and system->agent (role=='user')
    - or function/tool call returns (role=='function'/'tool').
    """

    def __init__(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        role: str,
        text: str,
        model: Optional[str] = None,  # model used to make function call
        name: Optional[str] = None,  # optional participant name
        created_at: Optional[datetime] = None,
        tool_calls: Optional[List[ToolCall]] = None,  # list of tool calls requested
        tool_call_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_dim: Optional[int] = None,
        embedding_model: Optional[str] = None,
        id: Optional[uuid.UUID] = None,
    ):
        super().__init__(id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.model = model  # model name (e.g. gpt-4)
        self.created_at = created_at if created_at is not None else datetime.now()

        # openai info
        assert role in ["system", "assistant", "user", "tool"]
        self.role = role  # role (agent/user/function)
        self.name = name

        # pad and store embeddings
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        self.embedding = (
            np.pad(embedding, (0, MAX_EMBEDDING_DIM - embedding.shape[0]), mode="constant").tolist() if embedding is not None else None
        )
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        if self.embedding is not None:
            assert self.embedding_dim, f"Must specify embedding_dim if providing an embedding"
            assert self.embedding_model, f"Must specify embedding_model if providing an embedding"
            assert len(self.embedding) == MAX_EMBEDDING_DIM, f"Embedding must be of length {MAX_EMBEDDING_DIM}"

        # tool (i.e. function) call info (optional)

        # if role == "assistant", this MAY be specified
        # if role != "assistant", this must be null
        assert tool_calls is None or isinstance(tool_calls, list)
        self.tool_calls = tool_calls

        # if role == "tool", then this must be specified
        # if role != "tool", this must be null
        if role == "tool":
            assert tool_call_id is not None
        else:
            assert tool_call_id is None
        self.tool_call_id = tool_call_id

    @staticmethod
    def dict_to_message(
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        openai_message_dict: dict,
        model: Optional[str] = None,  # model used to make function call
        created_at: Optional[datetime] = None,
    ):
        """Convert a ChatCompletion message object into a Message object (synced to DB)"""

        assert "role" in openai_message_dict, openai_message_dict
        assert "content" in openai_message_dict, openai_message_dict

        # Basic sanity check
        if openai_message_dict["role"] == "tool":
            assert "tool_call_id" in openai_message_dict and openai_message_dict["tool_call_id"] is not None, openai_message_dict
        else:
            if "tool_call_id" in openai_message_dict:
                assert openai_message_dict["tool_call_id"] is None, openai_message_dict

        if "tool_calls" in openai_message_dict and openai_message_dict["tool_calls"] is not None:
            assert openai_message_dict["role"] == "assistant", openai_message_dict

            tool_calls = [
                ToolCall(id=tool_call["id"], tool_call_type=tool_call["type"], function=tool_call["function"])
                for tool_call in openai_message_dict["tool_calls"]
            ]
        else:
            tool_calls = None

        # If we're going from tool-call style
        return Message(
            created_at=created_at,
            user_id=user_id,
            agent_id=agent_id,
            model=model,
            # standard fields expected in an OpenAI ChatCompletion message object
            role=openai_message_dict["role"],
            text=openai_message_dict["content"],
            name=openai_message_dict["name"] if "name" in openai_message_dict else None,
            tool_calls=tool_calls,
            tool_call_id=openai_message_dict["tool_call_id"] if "tool_call_id" in openai_message_dict else None,
        )

    def readable_message(self) -> Optional[str]:
        if self.role == "user":
            self_text_d = json.loads(self.text)
            if self_text_d.get("type") in ["login", "heartbeat"] or NON_USER_MSG_PREFIX in self_text_d["message"]:
                return None
            else:
                return self_text_d["message"]

        elif self.role == "tool":
            return None
        elif self.role == "assistant":
            if self.tool_calls:
                for tool_call in self.tool_calls:
                    if tool_call.function["name"] == "send_message":
                        try:
                            return json.loads(tool_call.function["arguments"], strict=False)["message"]
                        except json.JSONDecodeError:
                            logging.warning("Could not decode JSON, returning raw response.")
                            return tool_call.function["arguments"]
            elif "system alert" in self.text:
                pass
            else:
                logging.warning(f"Unexpected assistant message: role = {self.role}, text = {self.text}")
                pass

    def is_system_status_message(self) -> bool:
        return self.readable_message() is None

    def to_openai_dict(self):
        """Go from Message class to ChatCompletion message object"""

        # TODO change to pydantic casting, eg `return SystemMessageModel(self)`

        if self.role == "system":
            assert all([v is not None for v in [self.role]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional field, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name

        elif self.role == "user":
            assert all([v is not None for v in [self.text, self.role]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional field, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name

        elif self.role == "assistant":
            assert self.tool_calls is not None or self.text is not None
            openai_message = {
                "content": self.text,
                "role": self.role,
            }
            # Optional fields, do not include if null
            if self.name is not None:
                openai_message["name"] = self.name
            if self.tool_calls is not None:
                openai_message["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]  # type: ignore

        elif self.role == "tool":
            assert all([v is not None for v in [self.role, self.tool_call_id]]), vars(self)
            openai_message = {
                "content": self.text,
                "role": self.role,
                "tool_call_id": self.tool_call_id,
            }
        else:
            raise ValueError(self.role)

        return openai_message


class Passage(Record):
    """A passage is a single unit of memory, and a standard format accross all storage backends.

    It is a string of text with an assoidciated embedding.
    """

    def __init__(
        self,
        text: str,
        user_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,  # set if contained in agent memory
        embedding: Optional[np.ndarray] = None,
        embedding_dim: Optional[int] = None,
        embedding_model: Optional[str] = None,
        data_source: Optional[str] = None,  # None if created by agent
        doc_id: Optional[uuid.UUID] = None,
        id: Optional[uuid.UUID] = None,
        metadata_: Optional[dict] = {},
    ):
        if id is None:
            # by default, generate ID as a hash of the text (avoid duplicates)
            # TODO: use source-id instead?
            if agent_id:
                self.id = create_uuid_from_string("".join([text, str(agent_id), str(user_id)]))
            else:
                self.id = create_uuid_from_string("".join([text, str(user_id)]))
        else:
            self.id = id
        super().__init__(self.id)
        self.user_id = user_id
        self.agent_id = agent_id
        self.text = text
        self.data_source = data_source
        self.doc_id = doc_id
        self.metadata_ = metadata_

        # pad and store embeddings
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        self.embedding = (
            np.pad(embedding, (0, MAX_EMBEDDING_DIM - embedding.shape[0]), mode="constant").tolist() if embedding is not None else None
        )
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        if self.embedding is not None:
            assert self.embedding_dim, f"Must specify embedding_dim if providing an embedding"
            assert self.embedding_model, f"Must specify embedding_model if providing an embedding"
            assert len(self.embedding) == MAX_EMBEDDING_DIM, f"Embedding must be of length {MAX_EMBEDDING_DIM}"

        assert isinstance(self.user_id, uuid.UUID), f"UUID {self.user_id} must be a UUID type"
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert not agent_id or isinstance(self.agent_id, uuid.UUID), f"UUID {self.agent_id} must be a UUID type"
        assert not doc_id or isinstance(self.doc_id, uuid.UUID), f"UUID {self.doc_id} must be a UUID type"


class User:
    """Defines user and default configurations"""

    # TODO: make sure to encrypt/decrypt keys before storing in DB

    def __init__(
        self,
        id: uuid.UUID,
        # other
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"


class AgentState:
    def __init__(
        self,
        name: str,
        user_id: uuid.UUID,
        human: str,  # the filename where the human was originally sourced from
        id: Optional[uuid.UUID] = None,
        state: Optional[dict] = None,
        created_at: Optional[datetime] = None,
    ):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
        assert isinstance(self.id, uuid.UUID), f"UUID {self.id} must be a UUID type"
        assert isinstance(user_id, uuid.UUID), f"UUID {user_id} must be a UUID type"

        self.name = name
        self.user_id = user_id
        self.human = human

        self.created_at = created_at if created_at is not None else datetime.now()

        # state
        self.state = {} if not state else state


class Preset(BaseModel):
    name: str = Field(..., description="The name of the preset.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    created_at: datetime = Field(default_factory=datetime.now, description="The unix timestamp of when the preset was created.")
    human: str = Field(default=DEFAULT_HUMAN_TEXT, description="The human of the preset.")
