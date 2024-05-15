import uuid
from typing import Dict, List, Union, Optional, Tuple

from memgpt.data_types import AgentState, User, Preset, LLMConfig, EmbeddingConfig
from memgpt.config import MemGPTConfig
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.metadata import MetadataStore


def create_client() -> "Client":
    return Client()


class Client:
    def __init__(
        self,
        auto_save: bool = False,
        user_id: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.
        :param auto_save: indicates whether to automatically save after every message.
        :param debug: indicates whether to display debug messages.
        """
        self.auto_save = auto_save

        # determine user_id (pulled from local config)
        config = MemGPTConfig.load()
        if user_id:
            self.user_id = uuid.UUID(user_id)
        else:
            self.user_id = uuid.UUID(config.anon_clientid)

        # create user if does not exist
        ms = MetadataStore()
        self.user = User(id=self.user_id)
        if ms.get_user(self.user_id):
            # update user
            ms.update_user(self.user)
        else:
            ms.create_user(self.user)

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface=self.interface)

    def list_agents(self):
        self.interface.clear()
        return self.server.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existing = self.list_agents()
        if agent_id:
            return agent_id in [agent["id"] for agent in existing["agents"]]
        else:
            return agent_name in [agent["name"] for agent in existing["agents"]]

    def create_agent(
        self,
        name: str,
        preset: Optional[str] = None,
        human: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentState:
        if name and self.agent_exists(agent_name=name):
            raise ValueError(f"Agent with name {name} already exists (user_id={self.user_id})")

        self.interface.clear()
        agent_state = self.server.create_agent(
            user_id=self.user_id,
            name=name,
            preset=preset,
            human=human,
            embedding_config=embedding_config,
            llm_config=llm_config,
        )
        return agent_state

    def create_preset(self, preset: Preset):
        preset = self.server.create_preset(preset=preset)
        return preset

    def user_message(self, agent_id: str, message: str) -> Union[List[Dict], Tuple[List[Dict], int]]:  # type: ignore
        self.interface.clear()
        self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        else:
            return self.interface.to_list()

    def save(self):
        self.server.save_agents()
