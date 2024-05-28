import os
from dataclasses import dataclass
import configparser

from llama_index.embeddings.openai import OpenAIEmbedding

import memgpt.functions.function_sets.base


config = configparser.ConfigParser()
config_path = os.environ["MEMGPT_CONFIG_PATH"]

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

config.read(config_path)


@dataclass
class MemGPTConfig:
    function_modules = {m.strip() for m in config.get("functions", "modules").split(",")}.union(
        {memgpt.functions.function_sets.base.__name__}
    )

    # fmt: off
    # model parameters
    model = config.get("model", "model")
    model_endpoint_type = config.get("model", "model_endpoint_type")
    model_endpoint = config.get("model", "model_endpoint")

    embedding_endpoint = config.get("embedding", "embedding_endpoint")
    embedding_dim = int(config.get("embedding", "embedding_dim"))
    embedding_chunk_size = int(config.get("embedding", "embedding_chunk_size"))
    embedding_model_name = config.get("embedding", "embedding_model_name")

    embedding_model = OpenAIEmbedding(
        api_base=embedding_endpoint,  # type: ignore
        api_key=os.environ["OPENAI_API_KEY"],
    )
    # fmt: on
