""" Metadata store for user/agent/data_source information"""

import uuid
from typing import Optional

from memgpt.constants import ENGINE, SESSION_MAKER
from memgpt.utils import enforce_types
from memgpt.data_types import AgentState, User


from sqlalchemy import Column, JSON, DateTime, TypeDecorator, CHAR
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# Custom UUID type
class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(UUID(as_uuid=True))

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


class UserModel(Base):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)

    def __repr__(self) -> str:
        return f"<User(id='{self.id}')>"

    def to_record(self) -> User:
        return User(
            id=self.id,  # type: ignore
        )


class AgentModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # state
    state = Column(JSON)

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> AgentState:
        return AgentState(
            id=self.id,  # type: ignore
            user_id=self.user_id,  # type: ignore
            created_at=self.created_at,  # type: ignore
            state=self.state,  # type: ignore
        )


Base.metadata.create_all(
    ENGINE,
    tables=[
        UserModel.__table__,
        AgentModel.__table__,
    ],
)


class MetadataStore:
    @enforce_types
    def create_agent(self, agent: AgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        with SESSION_MAKER() as session:
            session.add(AgentModel(**vars(agent)))
            session.commit()

    @enforce_types
    def create_user(self, user: User):
        with SESSION_MAKER() as session:
            if session.query(UserModel).filter(UserModel.id == user.id).count() > 0:
                raise ValueError(f"User with id {user.id} already exists")
            session.add(UserModel(**vars(user)))
            session.commit()

    @enforce_types
    def update_agent(self, agent: AgentState):
        with SESSION_MAKER() as session:
            session.query(AgentModel).filter(AgentModel.id == agent.id).update(vars(agent))  # type: ignore
            session.commit()

    @enforce_types
    def update_user(self, user: User):
        with SESSION_MAKER() as session:
            session.query(UserModel).filter(UserModel.id == user.id).update(vars(user))  # type: ignore
            session.commit()

    @enforce_types
    def get_agent(self, agent_id: Optional[uuid.UUID] = None, user_id: Optional[uuid.UUID] = None) -> Optional[AgentState]:
        with SESSION_MAKER() as session:
            if agent_id:
                results = session.query(AgentModel).filter(AgentModel.id == agent_id).all()
            else:
                results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()

            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def get_user(self, user_id: uuid.UUID) -> Optional[User]:
        with SESSION_MAKER() as session:
            results = session.query(UserModel).filter(UserModel.id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()
