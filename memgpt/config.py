import dotenv
import yaml
import dotenv
import yaml
import os
import uuid


from memgpt.constants import MEMGPT_DIR
from memgpt.data_types import LLMConfig, EmbeddingConfig

dotenv.load_dotenv()

yaml_file = os.getenv("MEMGPT_CONFIG_PATH")

if yaml_file is None:
    raise ValueError("No config file found. Please set the MEMGPT_CONFIG_PATH environment variable.")

with open(yaml_file, "r") as file:
    config_data = yaml.safe_load(file)


class MemGPTConfig:
    storage_type: str = config_data["storage_type"]
    storage_uri_env: str = config_data["storage_uri_env"]

    config_path: str = os.path.join(MEMGPT_DIR, "config")
    anon_clientid: str = config_data["anon_clientid"]

    # preset
    preset = config_data["preset"]

    # persona parameters
    persona = config_data["persona"]
    human = config_data["human"]

    # model parameters
    default_llm_config = LLMConfig(**config_data["default_llm_config"])

    # embedding parameters
    default_embedding_config = EmbeddingConfig(**config_data["default_embedding_config"])

    storage_uri = os.getenv(config_data["storage_uri_env"])

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
    persistence_manager_type: str = config_data["storage_type"]
    persistence_manager_uri: str = storage_uri

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
