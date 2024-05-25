""" These classes define storage connectors.

We originally tried to use Llama Index VectorIndex, but their limited API was extremely problematic.
"""

import json
import logging
from typing import Optional, List, Union, Type
import uuid

from typing import List, Optional, Dict


from sqlalchemy import Column, String, BIGINT, JSON, DateTime, TypeDecorator, CHAR, select
from sqlalchemy.orm import mapped_column, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, insert
from sqlalchemy_json import MutableJson
import uuid

from pgvector.sqlalchemy import Vector
from memgpt.data_types import Message, Passage, ToolCall, Record, Passage, Message, RecordType
from memgpt.constants import ENGINE, MAX_EMBEDDING_DIM, NON_USER_MSG_PREFIX, SESSION_MAKER

from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func

RECALL_TABLE_NAME = "memgpt_recall_memory_agent"  # agent memory
ARCHIVAL_TABLE_NAME = "memgpt_archival_memory_agent"  # agent memory


Base = declarative_base()


# ENUM representing table types in MemGPT
# each table corresponds to a different table schema  (specified in data_types.py)
class TableType:
    ARCHIVAL_MEMORY = "archival_memory"  # recall memory table: memgpt_agent_{agent_id}
    RECALL_MEMORY = "recall_memory"  # archival memory table: memgpt_agent_recall_{agent_id}


# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


# Custom serialization / de-serialization for JSON columns


class ToolCallColumn(TypeDecorator):
    """Custom type for storing List[ToolCall] as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return [vars(v) for v in value]
        return value

    def process_result_value(self, value, dialect):
        if value:
            return [ToolCall(**v) for v in value]
        return value


class RecallMemoryModel(Base):
    """Defines data model for storing Message objects"""

    __tablename__ = RECALL_TABLE_NAME

    # Assuming message_id is the primary key
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(CommonUUID, nullable=False)
    agent_id = Column(CommonUUID, nullable=False)

    # openai info
    role = Column(String, nullable=False)
    text = Column(String)  # optional: can be null if function call
    model = Column(String)  # optional: can be null if LLM backend doesn't require specifying
    name = Column(String)  # optional: multi-agent only

    # tool call request info
    # TODO align with OpenAI spec of multiple tool calls
    tool_calls = Column(ToolCallColumn)

    # tool call response info
    # if role == "tool", then this must be specified
    # if role != "tool", this must be null
    tool_call_id = Column(String)

    # vector storage

    embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    embedding_dim = Column(BIGINT)
    embedding_model = Column(String)

    # Add a datetime column, with default value as the current time
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def is_system_status_message(self) -> bool:
        return self.readable_message() is None

    def readable_message(self) -> Optional[str]:
        if self.role == "user":  # type: ignore
            self_text_d = json.loads(self.text)  # type: ignore
            if self_text_d.get("type") in ["login", "heartbeat"] or NON_USER_MSG_PREFIX in self_text_d["message"]:
                return None
            else:
                return self_text_d["message"]

        elif self.role == "tool":  # type: ignore
            return None
        elif self.role == "assistant":  # type: ignore
            if self.tool_calls:  # type: ignore
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
                logging.warning(f"Unexpected assistant message: {self}")
                pass

    def __repr__(self):
        return f"<Message(message_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

    def to_record(self):
        return Message(
            user_id=self.user_id,  # type: ignore
            agent_id=self.agent_id,  # type: ignore
            role=self.role,  # type: ignore
            name=self.name,  # type: ignore
            text=self.text,  # type: ignore
            model=self.model,  # type: ignore
            tool_calls=self.tool_calls,  # type: ignore
            tool_call_id=self.tool_call_id,  # type: ignore
            embedding=self.embedding,
            embedding_dim=self.embedding_dim,  # type: ignore
            embedding_model=self.embedding_model,  # type: ignore
            created_at=self.created_at,  # type: ignore
            id=self.id,  # type: ignore
        )


class ArchivalMemoryModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = ARCHIVAL_TABLE_NAME

    # Assuming passage_id is the primary key
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    # id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(CommonUUID, nullable=False)
    text = Column(String)
    doc_id = Column(CommonUUID)
    agent_id = Column(CommonUUID)
    data_source = Column(String)  # agent_name if agent, data_source name if from data source

    # vector storage
    embedding = mapped_column(Vector(MAX_EMBEDDING_DIM))
    embedding_dim = Column(BIGINT)
    embedding_model = Column(String)

    metadata_ = Column(MutableJson)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Passage(passage_id='{self.id}', text='{self.text}', embedding='{self.embedding})>"

    def to_record(self):
        return Passage(
            text=self.text,  # type: ignore
            embedding=self.embedding,
            embedding_dim=self.embedding_dim,  # type: ignore
            embedding_model=self.embedding_model,  # type: ignore
            doc_id=self.doc_id,  # type: ignore
            user_id=self.user_id,  # type: ignore
            id=self.id,  # type: ignore
            data_source=self.data_source,  # type: ignore
            agent_id=self.agent_id,  # type: ignore
            metadata_=self.metadata_,  # type: ignore
        )


class StorageConnector:
    """Defines a DB connection that is user-specific to access data: Documents, Passages, Archival/Recall Memory"""

    type: Type[Record]

    def __init__(
        self,
        table_type: Union[TableType.ARCHIVAL_MEMORY, TableType.RECALL_MEMORY],  # type: ignore
        user_id,
        agent_id=None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.table_type = table_type

        # get object type
        if table_type == TableType.ARCHIVAL_MEMORY:
            self.type = Passage
            self.table_name = ARCHIVAL_TABLE_NAME
            self.db_model = ArchivalMemoryModel
        elif table_type == TableType.RECALL_MEMORY:
            self.type = Message
            self.table_name = RECALL_TABLE_NAME
            self.db_model = RecallMemoryModel
        else:
            raise ValueError(f"Table type {table_type} not implemented")
        logging.debug(f"Using table name {self.table_name}")

        # setup base filters for agent-specific tables
        if self.table_type == TableType.ARCHIVAL_MEMORY or self.table_type == TableType.RECALL_MEMORY:
            # agent-specific table
            assert agent_id is not None, "Agent ID must be provided for agent-specific tables"
            self.filters = {"user_id": self.user_id, "agent_id": self.agent_id}
        else:
            raise ValueError(f"Table type {table_type} not implemented")

    @staticmethod
    def get_archival_storage_connector(user_id, agent_id):
        return StorageConnector(TableType.ARCHIVAL_MEMORY, user_id, agent_id)

    @staticmethod
    def get_recall_storage_connector(user_id, agent_id):
        return StorageConnector(TableType.RECALL_MEMORY, user_id, agent_id)

    def get_filters(self, filters: Optional[Dict] = {}):
        if filters is not None:
            filter_conditions = {**self.filters, **filters}
        else:
            filter_conditions = self.filters
        all_filters = [getattr(self.db_model, key) == value for key, value in filter_conditions.items()]
        return all_filters

    def get_all(self, filters: Optional[Dict] = {}, limit=None) -> List[Record]:
        filters = self.get_filters(filters)  # type: ignore
        with SESSION_MAKER() as session:
            if limit:
                db_records = session.query(self.db_model).filter(*filters).limit(limit).all()  # type: ignore
            else:
                db_records = session.query(self.db_model).filter(*filters).all()  # type: ignore
        return [record.to_record() for record in db_records]

    def get(self, id: uuid.UUID) -> Optional[Record]:
        with SESSION_MAKER() as session:
            db_record = session.get(self.db_model, id)
        if db_record is None:
            return None
        return db_record.to_record()

    def size(self, filters: Optional[Dict] = {}) -> int:
        # return size of table
        filters = self.get_filters(filters)  # type: ignore
        with SESSION_MAKER() as session:
            return session.query(self.db_model).filter(*filters).count()  # type: ignore

    def insert(self, record: Record, exists_ok=True):
        self.insert_many([record], exists_ok=exists_ok)

    def insert_many(self, records: List[RecordType], exists_ok=True):

        # TODO: this is terrible, should eventually be done the same way for all types (migrate to SQLModel)
        if len(records) == 0:
            return
        if isinstance(records[0], Passage):
            with ENGINE.connect() as conn:
                db_records = [vars(record) for record in records]
                # print("records", db_records)
                stmt = insert(self.db_model.__table__).values(db_records)
                # print(stmt)
                if exists_ok:
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["id"], set_={c.name: c for c in stmt.excluded}  # Replace with your primary key column
                    )
                    print(upsert_stmt)
                    conn.execute(upsert_stmt)
                else:
                    conn.execute(stmt)
                conn.commit()
        else:
            with SESSION_MAKER() as session:
                for record in records:
                    db_record = self.db_model(**vars(record))
                    session.add(db_record)
                session.commit()

    def query(self, query: str, query_vec: List[float], top_k: int = 10, filters: Optional[Dict] = {}) -> List[Record]:
        filters = self.get_filters(filters)  # type: ignore
        with SESSION_MAKER() as session:
            results = session.scalars(
                select(self.db_model).filter(*filters).order_by(self.db_model.embedding.l2_distance(query_vec)).limit(top_k)  # type: ignore
            ).all()

        # Convert the results into Passage objects
        records = [result.to_record() for result in results]
        return records  # type: ignore

    def query_date(self, start_date, end_date, offset=0, limit=None):
        filters = self.get_filters({})
        with SESSION_MAKER() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(self.db_model.created_at >= start_date)
                .filter(self.db_model.created_at <= end_date)
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return [result.to_record() for result in results]

    def query_text(self, query, offset=0, limit=None):
        # todo: make fuzz https://stackoverflow.com/questions/42388956/create-a-full-text-search-index-with-sqlalchemy-on-postgresql/42390204#42390204
        filters = self.get_filters({})
        with SESSION_MAKER() as session:
            query = (
                session.query(self.db_model)
                .filter(*filters)
                .filter(func.lower(self.db_model.text).contains(func.lower(query)))
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            results = query.all()
        # return [self.type(**vars(result)) for result in results]
        return [result.to_record() for result in results]

    def delete(self, filters: Optional[Dict] = {}):
        filters = self.get_filters(filters)  # type: ignore
        with SESSION_MAKER() as session:
            session.query(self.db_model).filter(*filters).delete()  # type: ignore
            session.commit()


Base.metadata.create_all(
    ENGINE,
    tables=[
        RecallMemoryModel.__table__,
        ArchivalMemoryModel.__table__,
    ],
)
