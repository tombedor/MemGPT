import os
import shutil
import uuid
from typing import Annotated, Optional
from enum import Enum
from typing import Annotated

import typer
from prettytable import PrettyTable
from tqdm import tqdm

from memgpt.config import MemGPTConfig, utils
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import MemGPTConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.data_types import Source
from memgpt.metadata import MetadataStore

app = typer.Typer()


def get_azure_credentials():
    creds = dict(
        azure_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    # embedding endpoint and version default to non-embedding
    creds["azure_embedding_endpoint"] = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", creds["azure_endpoint"])
    creds["azure_embedding_version"] = os.getenv("AZURE_OPENAI_EMBEDDING_VERSION", creds["azure_version"])
    return creds


def get_openai_credentials():
    openai_key = os.getenv("OPENAI_API_KEY")
    return openai_key


class ListChoice(str, Enum):
    agents = "agents"
    humans = "humans"
    personas = "personas"
    sources = "sources"


@app.command()
def list(arg: Annotated[ListChoice, typer.Argument]):
    ms = MetadataStore()
    user_id = uuid.UUID(MemGPTConfig.anon_clientid)
    if arg == ListChoice.agents:
        """List all agents"""
        table = PrettyTable()
        table.field_names = ["Name", "LLM Model", "Embedding Model", "Embedding Dim", "Persona", "Human", "Data Source", "Create Time"]
        for agent in tqdm(ms.list_agents(user_id=user_id)):
            source_ids = ms.list_attached_sources(agent_id=agent.id)
            assert all([source_id is not None and isinstance(source_id, uuid.UUID) for source_id in source_ids])
            sources = [ms.get_source(source_id=source_id) for source_id in source_ids]
            assert all([source is not None and isinstance(source, Source)] for source in sources)
            source_names = [source.name for source in sources if source is not None]
            table.add_row(
                [
                    agent.name,
                    agent.llm_config.model,
                    agent.embedding_config.embedding_model,
                    agent.embedding_config.embedding_dim,
                    agent.persona,
                    agent.human,
                    ",".join(source_names),
                    utils.format_datetime(agent.created_at),
                ]
            )
        print(table)
    elif arg == ListChoice.humans:
        """List all humans"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for human_file in utils.list_human_files():
            text = open(human_file, "r").read()
            name = os.path.basename(human_file).replace("txt", "")
            table.add_row([name, text])
        print(table)
    elif arg == ListChoice.personas:
        """List all personas"""
        table = PrettyTable()
        table.field_names = ["Name", "Text"]
        for persona_file in utils.list_persona_files():
            print(persona_file)
            text = open(persona_file, "r").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            table.add_row([name, text])
        print(table)
    elif arg == ListChoice.sources:
        """List all data sources"""

        # create table
        table = PrettyTable()
        table.field_names = ["Name", "Embedding Model", "Embedding Dim", "Created At", "Agents"]
        # TODO: eventually look accross all storage connections
        # TODO: add data source stats
        # TODO: connect to agents

        # get all sources
        for source in ms.list_sources(user_id=user_id):
            # get attached agents
            agent_ids = ms.list_attached_agents(source_id=source.id)
            agent_states = [ms.get_agent(agent_id=agent_id) for agent_id in agent_ids]
            agent_names = [agent_state.name for agent_state in agent_states if agent_state is not None]

            table.add_row(
                [source.name, source.embedding_model, source.embedding_dim, utils.format_datetime(source.created_at), ",".join(agent_names)]
            )

        print(table)
    else:
        raise ValueError(f"Unknown argument {arg}")


@app.command()
def add(
    option: str,  # [human, persona]
    name: Annotated[str, typer.Option(help="Name of human/persona")],
    text: Annotated[Optional[str], typer.Option(help="Text of human/persona")] = None,
    filename: Annotated[Optional[str], typer.Option("-f", help="Specify filename")] = None,
):
    """Add a person/human"""

    if option == "persona":
        directory = os.path.join(MEMGPT_DIR, "personas")
    elif option == "human":
        directory = os.path.join(MEMGPT_DIR, "humans")
    else:
        raise ValueError(f"Unknown kind {option}")

    if filename:
        assert text is None, f"Cannot provide both filename and text"
        # copy file to directory
        shutil.copyfile(filename, os.path.join(directory, name))
    if text:
        assert filename is None, f"Cannot provide both filename and text"
        # write text to file
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(text)


@app.command()
def delete(option: str, name: str):
    """Delete a source from the archival memory."""

    user_id = uuid.UUID(MemGPTConfig.anon_clientid)
    ms = MetadataStore()
    assert ms.get_user(user_id=user_id), f"User {user_id} does not exist"

    try:
        # delete from metadata
        if option == "source":
            # delete metadata
            source = ms.get_source(source_name=name, user_id=user_id)
            assert source is not None, f"Source {name} does not exist"
            ms.delete_source(source_id=source.id)

            # delete from passages
            conn = StorageConnector.get_storage_connector(TableType.PASSAGES, user_id=user_id)
            conn.delete({"data_source": name})

            assert (
                conn.get_all({"data_source": name}) == []
            ), f"Expected no passages with source {name}, but got {conn.get_all({'data_source': name})}"

            # TODO: should we also delete from agents?
        elif option == "agent":
            agent = ms.get_agent(agent_name=name, user_id=user_id)
            assert agent is not None, f"Agent {name} for user_id {user_id} does not exist"

            # recall memory
            recall_conn = StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, user_id=user_id, agent_id=agent.id)
            recall_conn.delete({"agent_id": agent.id})

            # archival memory
            archival_conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, user_id=user_id, agent_id=agent.id)
            archival_conn.delete({"agent_id": agent.id})

            # metadata
            ms.delete_agent(agent_id=agent.id)

        else:
            raise ValueError(f"Option {option} not implemented")

        typer.secho(f"Deleted source '{name}'", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Failed to deleted source '{name}'\n{e}", fg=typer.colors.RED)
