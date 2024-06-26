""" Metadata store for user/agent/data_source information"""

import uuid
from typing import Optional

from memgpt.agent_store.storage import AgentModel, UserModel
from memgpt.constants import SESSION_MAKER
from memgpt.utils import enforce_types
from memgpt.data_types import AgentState, User


from sqlalchemy.orm import declarative_base

Base = declarative_base()


# Custom UUID type


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
