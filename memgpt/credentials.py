import os
from dataclasses import dataclass
import configparser
from typing import Optional


from memgpt.constants import MEMGPT_DIR


SUPPORTED_AUTH_TYPES = ["bearer_token", "api_key"]

# helper functions for writing to configs
def get_field(config, section, field):
    if section not in config:
        return None
    if config.has_option(section, field):
        return config.get(section, field)
    else:
        return None


def set_field(config, section, field, value):
    if value is None:  # cannot write None
        return
    if section not in config:  # create section
        config.add_section(section)
    config.set(section, field, value)


@dataclass
class MemGPTCredentials:
    # credentials for MemGPT
    credentials_path: str = os.path.join(MEMGPT_DIR, "credentials")

    # openai config
    openai_auth_type: str = "bearer_token"
    openai_key: Optional[str] = None

    # azure config
    azure_auth_type: str = "api_key"
    azure_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_version: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None

    # custom llm API config
    openllm_auth_type: Optional[str] = None
    openllm_key: Optional[str] = None

    @classmethod
    def load(cls) -> "MemGPTCredentials":
        config = configparser.ConfigParser()

        # allow overriding with env variables
        if os.getenv("MEMGPT_CREDENTIALS_PATH"):
            credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        else:
            credentials_path = MemGPTCredentials.credentials_path

        if os.path.exists(credentials_path):
            # read existing credentials
            config.read(credentials_path)
            config_dict = {
                # openai
                "openai_auth_type": get_field(config, "openai", "auth_type"),
                "openai_key": get_field(config, "openai", "key"),
                # azure
                "azure_auth_type": get_field(config, "azure", "auth_type"),
                "azure_key": get_field(config, "azure", "key"),
                "azure_endpoint": get_field(config, "azure", "endpoint"),
                "azure_version": get_field(config, "azure", "version"),
                "azure_deployment": get_field(config, "azure", "deployment"),
                "azure_embedding_deployment": get_field(config, "azure", "embedding_deployment"),
                # open llm
                "openllm_auth_type": get_field(config, "openllm", "auth_type"),
                "openllm_key": get_field(config, "openllm", "key"),
                # path
                "credentials_path": credentials_path,
            }
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            return cls(**config_dict)

        # create new config
        config = cls(credentials_path=credentials_path)
        config.save()  # save updated config
        return config

    def save(self):
        import memgpt

        config = configparser.ConfigParser()
        # openai config
        set_field(config, "openai", "auth_type", self.openai_auth_type)
        set_field(config, "openai", "key", self.openai_key)

        # azure config
        set_field(config, "azure", "auth_type", self.azure_auth_type)
        set_field(config, "azure", "key", self.azure_key)
        set_field(config, "azure", "endpoint", self.azure_endpoint)
        set_field(config, "azure", "version", self.azure_version)
        set_field(config, "azure", "deployment", self.azure_deployment)
        set_field(config, "azure", "embedding_deployment", self.azure_embedding_deployment)

        # openai config
        set_field(config, "openllm", "auth_type", self.openllm_auth_type)
        set_field(config, "openllm", "key", self.openllm_key)

        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)
        with open(self.credentials_path, "w", encoding="utf-8") as f:
            config.write(f)

    @staticmethod
    def exists():
        # allow overriding with env variables
        if os.getenv("MEMGPT_CREDENTIALS_PATH"):
            credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        else:
            credentials_path = MemGPTCredentials.credentials_path

        assert not os.path.isdir(credentials_path), f"Credentials path {credentials_path} cannot be set to a directory."
        return os.path.exists(credentials_path)
