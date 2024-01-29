import dotenv
import yaml
import inspect
import json
import os
import uuid
from dataclasses import dataclass

import memgpt.utils as utils

from memgpt.constants import MEMGPT_DIR
from memgpt.data_types import AgentState, LLMConfig, EmbeddingConfig

dotenv.load_dotenv()

yaml_file = os.getenv("MEMGPT_CONFIG_PATH")

if yaml_file is None:
    raise ValueError("No config file found. Please set the MEMGPT_CONFIG_PATH environment variable.")

with open(yaml_file, "r") as file:
    config_data = yaml.safe_load(file)


class MemGPTConfig:
    storage_type = config_data["storage_type"]
    storage_uri = os.getenv(config_data["storage_uri_env"])

    anon_clientid = config_data["anon_clientid"]

    # preset
    preset = config_data["preset"]

    # persona parameters
    persona = config_data["persona"]
    human = config_data["human"]

    # model parameters
    default_llm_config = LLMConfig(**config_data["default_llm_config"])

    # embedding parameters
    default_embedding_config = EmbeddingConfig(**config_data["default_embedding_config"])

    # database configs: archival
    archival_storage_type = config_data["storage_type"]
    archival_storage_uri = storage_uri

    # database configs: recall
    recall_storage_type = config_data["storage_type"]
    recall_storage_uri = storage_uri

    # database configs: metadata storage (sources, agents, data sources)
    metadata_storage_type = config_data["storage_type"]
    metadata_storage_uri = storage_uri

    # database configs: agent state
    persistence_manager_type = config_data["storage_type"]
    persistence_manager_uri = storage_uri

    @staticmethod
    def generate_uuid() -> str:
        return uuid.UUID(int=uuid.getnode()).hex

    @staticmethod
    def create_config_dir():
        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)

        folders = ["personas", "humans", "archival", "agents", "functions", "system_prompts", "presets", "settings"]

        for folder in folders:
            if not os.path.exists(os.path.join(MEMGPT_DIR, folder)):
                os.makedirs(os.path.join(MEMGPT_DIR, folder))


@dataclass
class AgentConfig:
    """

    NOTE: this is a deprecated class, use AgentState instead. This class is only used for backcompatibility.
    Configuration for a specific instance of an agent
    """

    def __init__(
        self,
        persona,
        human,
        # model info
        model=None,
        model_endpoint_type=None,
        model_endpoint=None,
        model_wrapper=None,
        context_window=None,
        # embedding info
        embedding_endpoint_type=None,
        embedding_endpoint=None,
        embedding_model=None,
        embedding_dim=None,
        embedding_chunk_size=None,
        # other
        preset=None,
        data_sources=None,
        # agent info
        agent_config_path=None,
        name=None,
        create_time=None,
        memgpt_version=None,
        # functions
        functions=None,  # schema definitions ONLY (linked at runtime)
    ):
        if name is None:
            self.name = f"agent_{self.generate_agent_id()}"
        else:
            self.name = name

        self.persona = MemGPTConfig.persona if persona is None else persona
        self.human = MemGPTConfig.human if human is None else human
        self.preset = MemGPTConfig.preset if preset is None else preset
        self.context_window = MemGPTConfig.default_llm_config.context_window if context_window is None else context_window
        self.model = MemGPTConfig.default_llm_config.model if model is None else model
        self.model_endpoint_type = (
            MemGPTConfig.default_llm_config.model_endpoint_type if model_endpoint_type is None else model_endpoint_type
        )
        self.model_endpoint = MemGPTConfig.default_llm_config.model_endpoint if model_endpoint is None else model_endpoint
        self.model_wrapper = MemGPTConfig.default_llm_config.model_wrapper if model_wrapper is None else model_wrapper
        self.llm_config = LLMConfig(
            model=self.model,
            model_endpoint_type=self.model_endpoint_type,
            model_endpoint=self.model_endpoint,
            model_wrapper=self.model_wrapper,
            context_window=self.context_window,
        )
        self.embedding_endpoint_type = (
            MemGPTConfig.default_embedding_config.embedding_endpoint_type if embedding_endpoint_type is None else embedding_endpoint_type
        )
        self.embedding_endpoint = (
            MemGPTConfig.default_embedding_config.embedding_endpoint if embedding_endpoint is None else embedding_endpoint
        )
        self.embedding_model = MemGPTConfig.default_embedding_config.embedding_model if embedding_model is None else embedding_model
        self.embedding_dim = MemGPTConfig.default_embedding_config.embedding_dim if embedding_dim is None else embedding_dim
        self.embedding_chunk_size = (
            MemGPTConfig.default_embedding_config.embedding_chunk_size if embedding_chunk_size is None else embedding_chunk_size
        )
        self.embedding_config = EmbeddingConfig(
            embedding_endpoint_type=self.embedding_endpoint_type,
            embedding_endpoint=self.embedding_endpoint,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
            embedding_chunk_size=self.embedding_chunk_size,
        )

        # agent metadata
        self.data_sources = data_sources if data_sources is not None else []
        self.create_time = create_time if create_time is not None else utils.get_local_time()
        if memgpt_version is None:
            import memgpt

            self.memgpt_version = memgpt.__version__
        else:
            self.memgpt_version = memgpt_version

        # functions
        self.functions = functions

        # save agent config
        self.agent_config_path = (
            os.path.join(MEMGPT_DIR, "agents", self.name, "config.json") if agent_config_path is None else agent_config_path
        )

    def generate_agent_id(self, length=6):
        ## random character based
        # characters = string.ascii_lowercase + string.digits
        # return ''.join(random.choices(characters, k=length))

        # count based
        agent_count = len(utils.list_agent_config_files())
        return str(agent_count + 1)

    def attach_data_source(self, data_source: str):
        # TODO: add warning that only once source can be attached
        # i.e. previous source will be overriden
        self.data_sources.append(data_source)
        self.save()

    def save_dir(self):
        return os.path.join(MEMGPT_DIR, "agents", self.name)

    def save_state_dir(self):
        # directory to save agent state
        return os.path.join(MEMGPT_DIR, "agents", self.name, "agent_state")

    def save_persistence_manager_dir(self):
        # directory to save persistent manager state
        return os.path.join(MEMGPT_DIR, "agents", self.name, "persistence_manager")

    def save_agent_index_dir(self):
        # save llama index inside of persistent manager directory
        return os.path.join(self.save_persistence_manager_dir(), "index")

    def save(self):
        # save state of persistence manager
        os.makedirs(os.path.join(MEMGPT_DIR, "agents", self.name), exist_ok=True)
        # save version
        self.memgpt_version = memgpt.__version__
        with open(self.agent_config_path, "w") as f:
            json.dump(vars(self), f, indent=4)

    def to_agent_state(self):
        return AgentState(
            name=self.name,
            preset=self.preset,
            persona=self.persona,
            human=self.human,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            create_time=self.create_time,
        )

    @staticmethod
    def exists(name: str):
        """Check if agent config exists"""
        agent_config_path = os.path.join(MEMGPT_DIR, "agents", name)
        return os.path.exists(agent_config_path)

    @classmethod
    def load(cls, name: str):
        """Load agent config from JSON file"""
        agent_config_path = os.path.join(MEMGPT_DIR, "agents", name, "config.json")
        assert os.path.exists(agent_config_path), f"Agent config file does not exist at {agent_config_path}"
        with open(agent_config_path, "r") as f:
            agent_config = json.load(f)
        # allow compatibility accross versions
        try:
            class_args = inspect.getargspec(cls.__init__).args
        except AttributeError:
            # https://github.com/pytorch/pytorch/issues/15344
            class_args = inspect.getfullargspec(cls.__init__).args
        agent_fields = list(agent_config.keys())
        for key in agent_fields:
            if key not in class_args:
                utils.printd(f"Removing missing argument {key} from agent config")
                del agent_config[key]
        return cls(**agent_config)


if __name__ == "__main__":
    config = MemGPTConfig.load()
    print(config)
