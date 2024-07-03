import logging
import uuid
from typing import List


import memgpt.system as system
from memgpt.agent import Agent

from memgpt.data_types import (
    Message,
)

from memgpt.metadata import MetadataStore

logger = logging.getLogger(__name__)


MAX_CHAINING_STEPS = 20


class SyncServer:

    def __init__(
        self,
    ):

        # Initialize the metadata store
        self.ms = MetadataStore()

    def _load_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> Agent:
        """Loads a saved agent into memory (if it doesn't exist, throw an error)"""
        assert isinstance(user_id, uuid.UUID), user_id
        assert isinstance(agent_id, uuid.UUID), agent_id

        try:
            logger.debug(f"Grabbing agent user_id={user_id} agent_id={agent_id} from database")
            agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
            if not agent_state:
                logger.exception(f"agent_id {agent_id} does not exist")
                raise ValueError(f"agent_id {agent_id} does not exist")

            # Instantiate an agent object using the state retrieved
            logger.debug(f"Creating an agent object")
            memgpt_agent = Agent(agent_state=agent_state)

            return memgpt_agent

        except Exception as e:
            logger.exception(f"Error occurred while trying to get agent {agent_id}:\n{e}")
            raise

    def user_message(self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str) -> List[Message]:
        """Process an incoming user message and feed it through the MemGPT agent"""

        # Basic input sanitization
        if len(message) == 0:
            raise ValueError(f"Invalid input: '{message}'")

        packaged_user_message = system.package_user_message(user_message=message)

        agent = self._load_agent(user_id=user_id, agent_id=agent_id)

        # Run the agent state forward
        return agent.step_chain(input_message=packaged_user_message)
