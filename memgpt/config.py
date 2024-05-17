import os
from typing import Set
from dataclasses import dataclass, field
import configparser

import memgpt

import memgpt.functions.function_sets.base

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PRESET
from memgpt.data_types import LLMConfig, EmbeddingConfig


# helper functions for writing to configs
def get_field(config, section, field):
    if section not in config:
        return None
    if config.has_option(section, field):
        return config.get(section, field)
    else:
        return None


@dataclass
class MemGPTConfig:
    # preset
    preset: str = DEFAULT_PRESET

    # persona parameters
    human: str = DEFAULT_HUMAN

    # model parameters
    default_llm_config: LLMConfig = field(default_factory=LLMConfig)

    # embedding parameters
    default_embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # full names of modules
    functions_modules: Set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "MemGPTConfig":

        config = configparser.ConfigParser()

        # set by env var
        config_path = os.environ["MEMGPT_CONFIG_PATH"]

        # insure all configuration directories exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # read existing config
        config.read(config_path)

        # Handle extraction of nested LLMConfig and EmbeddingConfig
        llm_config_dict = {
            # Extract relevant LLM configuration from the config file
            "model": get_field(config, "model", "model"),
            "model_endpoint": get_field(config, "model", "model_endpoint"),
            "model_endpoint_type": get_field(config, "model", "model_endpoint_type"),
            "model_wrapper": get_field(config, "model", "model_wrapper"),
            "context_window": get_field(config, "model", "context_window"),
        }
        embedding_config_dict = {
            # Extract relevant Embedding configuration from the config file
            "embedding_endpoint": get_field(config, "embedding", "embedding_endpoint"),
            "embedding_model": get_field(config, "embedding", "embedding_model"),
            "embedding_endpoint_type": get_field(config, "embedding", "embedding_endpoint_type"),
            "embedding_dim": get_field(config, "embedding", "embedding_dim"),
            "embedding_chunk_size": get_field(config, "embedding", "embedding_chunk_size"),
        }
        # Remove null values
        llm_config_dict = {k: v for k, v in llm_config_dict.items() if v is not None}
        embedding_config_dict = {k: v for k, v in embedding_config_dict.items() if v is not None}
        # Correct the types that aren't strings
        if llm_config_dict["context_window"] is not None:
            llm_config_dict["context_window"] = int(llm_config_dict["context_window"])  # type: ignore
        if embedding_config_dict["embedding_dim"] is not None:
            embedding_config_dict["embedding_dim"] = int(embedding_config_dict["embedding_dim"])  # type: ignore
        if embedding_config_dict["embedding_chunk_size"] is not None:
            embedding_config_dict["embedding_chunk_size"] = int(embedding_config_dict["embedding_chunk_size"])  # type: ignore
        # Construct the inner properties
        llm_config = LLMConfig(**llm_config_dict)  # type: ignore
        embedding_config = EmbeddingConfig(**embedding_config_dict)  # type: ignore

        if config.has_option("functions", "modules"):
            function_modules = {m.strip() for m in config.get("functions", "modules").split(",")}
        else:
            function_modules = set()
        function_modules.add(memgpt.functions.function_sets.base.__name__)

        # Everything else
        config_dict = {
            # Two prepared configs
            "default_llm_config": llm_config,
            "default_embedding_config": embedding_config,
            # Agent related
            "preset": get_field(config, "defaults", "preset"),
            "human": get_field(config, "defaults", "human"),
            # Storage related
            # Misc
            "functions_modules": function_modules,
        }

        # Don't include null values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return cls(**config_dict)
