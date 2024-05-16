""" Metadata store for user/agent/data_source information"""

import uuid
from typing import Optional, List

from memgpt.constants import ENGINE, SESSION_MAKER
from memgpt.utils import enforce_types
from memgpt.data_types import AgentState, User, LLMConfig, EmbeddingConfig, Preset
from memgpt.agent_store.storage import Base

from memgpt.models.pydantic_models import PersonaModel, HumanModel

from sqlalchemy import Column, String, JSON
from sqlalchemy import func
from sqlalchemy.sql import func
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import TypeDecorator, CHAR


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


class LLMConfigColumn(TypeDecorator):
    """Custom type for storing LLMConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return LLMConfig(**value)
        return value


class EmbeddingConfigColumn(TypeDecorator):
    """Custom type for storing EmbeddingConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
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
    name = Column(String, nullable=False)
    persona = Column(String)
    human = Column(String)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # configs
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    state = Column(JSON)

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> AgentState:
        return AgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            human=self.human,
            created_at=self.created_at,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            state=self.state,
        )


class PresetModel(Base):
    """Defines data model for storing Preset objects"""

    __tablename__ = "presets"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    human = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Preset(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Preset:
        return Preset(
            id=self.id,  # type: ignore
            user_id=self.user_id,  # type: ignore
            name=self.name,  # type: ignore
            description=self.description,  # type: ignore
            human=self.human,  # type: ignore
            created_at=self.created_at,  # type: ignore
        )
Base.metadata.create_all(
    ENGINE,
    tables=[
        UserModel.__table__,
        AgentModel.__table__,
        PresetModel.__table__,
        HumanModel.__table__,
        PersonaModel.__table__,
    ],
)

class MetadataStore:
    @enforce_types
    def create_agent(self, agent: AgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        with SESSION_MAKER() as session:
            if session.query(AgentModel).filter(AgentModel.name == agent.name).filter(AgentModel.user_id == agent.user_id).count() > 0:
                raise ValueError(f"Agent with name {agent.name} already exists")
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
    def create_preset(self, preset: Preset):
        with SESSION_MAKER() as session:
            if session.query(PresetModel).filter(PresetModel.id == preset.id).count() > 0:
                raise ValueError(f"User with id {preset.id} already exists")
            session.add(PresetModel(**vars(preset)))
            session.commit()

    @enforce_types
    def get_preset(
        self, preset_id: Optional[uuid.UUID] = None, preset_name: Optional[str] = None, user_id: Optional[uuid.UUID] = None
    ) -> Optional[Preset]:
        with SESSION_MAKER() as session:
            if preset_id:
                results = session.query(PresetModel).filter(PresetModel.id == preset_id).all()
            elif preset_name and user_id:
                results = session.query(PresetModel).filter(PresetModel.name == preset_name).filter(PresetModel.user_id == user_id).all()
            else:
                raise ValueError("Must provide either preset_id or (preset_name and user_id)")
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

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
    def delete_agent(self, agent_id: uuid.UUID):
        with SESSION_MAKER() as session:
            session.query(AgentModel).filter(AgentModel.id == agent_id).delete()
            session.commit()

    @enforce_types
    def list_agents(self, user_id: uuid.UUID) -> List[AgentState]:
        with SESSION_MAKER() as session:
            results = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            return [r.to_record() for r in results]

    @enforce_types
    def get_agent(
        self, agent_id: Optional[uuid.UUID] = None, agent_name: Optional[str] = None, user_id: Optional[uuid.UUID] = None
    ) -> Optional[AgentState]:
        with SESSION_MAKER() as session:
            if agent_id:
                results = session.query(AgentModel).filter(AgentModel.id == agent_id).all()
            else:
                assert agent_name is not None and user_id is not None, "Must provide either agent_id or agent_name"
                results = session.query(AgentModel).filter(AgentModel.name == agent_name).filter(AgentModel.user_id == user_id).all()

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

    @enforce_types
    def add_human(self, human: HumanModel):
        with SESSION_MAKER() as session:
            session.add(human)
            session.commit()

    @enforce_types
    def add_persona(self, persona: PersonaModel):
        with SESSION_MAKER() as session:
            session.add(persona)
            session.commit()
