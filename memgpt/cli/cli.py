import uuid
import sys
import io
import logging
from pathlib import Path
import os
import subprocess
from enum import Enum
from typing import Annotated, Optional

import typer
import questionary
from llama_index import set_global_service_context
from llama_index import ServiceContext
from memgpt.config import MemGPTConfig

from memgpt.log import logger
from memgpt.interface import CLIInterface as interface  # for printing to terminal
import memgpt.presets.presets as presets
import memgpt.utils as utils
from memgpt.utils import printd, open_folder_in_explorer, suppress_stdout
from memgpt.constants import MEMGPT_DIR, CLI_WARNING_PREFIX
from memgpt.agent import Agent
from memgpt.embeddings import embedding_model
from memgpt.server.constants import WS_DEFAULT_PORT, REST_DEFAULT_PORT
from memgpt.data_types import AgentState, User, Passage
from memgpt.metadata import MetadataStore, save_agent


class QuickstartChoice(Enum):
    openai = "openai"
    # azure = "azure"
    memgpt_hosted = "memgpt"


def str_to_quickstart_choice(choice_str: str) -> QuickstartChoice:
    try:
        return QuickstartChoice[choice_str]
    except KeyError:
        valid_options = [choice.name for choice in QuickstartChoice]
        raise ValueError(f"{choice_str} is not a valid QuickstartChoice. Valid options are: {valid_options}")


def open_folder():
    """Open a folder viewer of the MemGPT home directory"""
    try:
        print(f"Opening home folder: {MEMGPT_DIR}")
        open_folder_in_explorer(MEMGPT_DIR)
    except Exception as e:
        print(f"Failed to open folder with system viewer, error:\n{e}")


class ServerChoice(Enum):
    rest_api = "rest"
    ws_api = "websocket"


def create_default_user_or_exit(ms: MetadataStore):
    user_id = uuid.UUID(MemGPTConfig.anon_clientid)
    user = ms.get_user(user_id=user_id)
    if user is None:
        ms.create_user(User(id=user_id))
        user = ms.get_user(user_id=user_id)
        if user is None:
            typer.secho(f"Failed to create default user in database.", fg=typer.colors.RED)
            sys.exit(1)
        else:
            return user
    else:
        return user


def server(
    type: Annotated[ServerChoice, typer.Option(help="Server to run")] = "rest",
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    debug: Annotated[bool, typer.Option(help="Turn debugging output on")] = True,
):
    """Launch a MemGPT server process"""

    if debug:
        from memgpt.server.server import logger as server_logger

        # Set the logging level
        server_logger.setLevel(logging.DEBUG)
        # Create a StreamHandler
        stream_handler = logging.StreamHandler()
        # Set the formatter (optional)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        # Add the handler to the logger
        server_logger.addHandler(stream_handler)

    if type == ServerChoice.rest_api:
        import uvicorn
        from memgpt.server.rest_api.server import app

        ms = MetadataStore()
        create_default_user_or_exit(ms)

        try:
            # Start the subprocess in a new session
            uvicorn.run(app, host=host or "localhost", port=port or REST_DEFAULT_PORT)

        except KeyboardInterrupt:
            # Handle CTRL-C
            typer.secho("Terminating the server...")
            sys.exit(0)

    elif type == ServerChoice.ws_api:
        if port is None:
            port = WS_DEFAULT_PORT

        # Change to the desired directory
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent

        server_directory = os.path.join(script_dir.parent, "server", "ws_api")
        command = f"python server.py {port}"

        # Run the command
        typer.secho(f"Running WS (websockets) server: {command} (inside {server_directory})")

        process = None
        try:
            # Start the subprocess in a new session
            process = subprocess.Popen(command, shell=True, start_new_session=True, cwd=server_directory)
            process.wait()
        except KeyboardInterrupt:
            # Handle CTRL-C
            if process is not None:
                typer.secho("Terminating the server...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    typer.secho("Server terminated with kill()")
            sys.exit(0)


def run(
    persona: Annotated[Optional[str], typer.Option(help="Specify persona")] = None,
    agent: Annotated[Optional[str], typer.Option(help="Specify agent save file")] = None,
    human: Annotated[Optional[str], typer.Option(help="Specify human")] = None,
    preset: Annotated[Optional[str], typer.Option(help="Specify preset")] = None,
    # model flags
    model: Annotated[Optional[str], typer.Option(help="Specify the LLM model")] = None,
    model_wrapper: Annotated[Optional[str], typer.Option(help="Specify the LLM model wrapper")] = None,
    model_endpoint: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint")] = None,
    model_endpoint_type: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint type")] = None,
    context_window: Annotated[
        Optional[int], typer.Option(help="The context window of the LLM you are using (e.g. 8k for most Mistral 7B variants)")
    ] = None,
    # other
    first: Annotated[bool, typer.Option(help="Use --first to send the first message in the sequence")] = False,
    strip_ui: Annotated[bool, typer.Option(help="Remove all the bells and whistles in CLI output (helpful for testing)")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    no_verify: Annotated[bool, typer.Option(help="Bypass message verification")] = False,
    yes: Annotated[bool, typer.Option("-y", help="Skip confirmation prompt and use defaults")] = False,
):
    """Start chatting with an MemGPT agent

    Example usage: `memgpt run --agent myagent --data-source mydata --persona mypersona --human myhuman --model gpt-3.5-turbo`

    :param persona: Specify persona
    :param agent: Specify agent name (will load existing state if the agent exists, or create a new one with that name)
    :param human: Specify human
    :param model: Specify the LLM model

    """

    # setup logger
    # TODO: remove Utils Debug after global logging is complete.
    utils.DEBUG = debug
    # TODO: add logging command line options for runtime log level

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # read user id from config
    ms = MetadataStore()
    user = create_default_user_or_exit(ms)

    # determine agent to use, if not provided
    if not yes and not agent:
        agents = ms.list_agents(user_id=user.id)
        agents = [a.name for a in agents]

        if len(agents) > 0 and not any([persona, human, model]):
            print()
            select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
            if select_agent is None:
                raise KeyboardInterrupt
            if select_agent:
                agent = questionary.select("Select agent:", choices=agents).ask()

    # create agent config
    agent_state = ms.get_agent(agent_name=agent, user_id=user.id) if agent else None
    if agent and agent_state:  # use existing agent
        typer.secho(f"\n🔁 Using existing agent {agent}", fg=typer.colors.GREEN)
        # agent_config = AgentConfig.load(agent)
        # agent_state = ms.get_agent(agent_name=agent, user_id=user_id)
        printd("Loading agent state:", agent_state.id)
        printd("Agent state:", agent_state.state)
        # printd("State path:", agent_config.save_state_dir())
        # printd("Persistent manager path:", agent_config.save_persistence_manager_dir())
        # printd("Index path:", agent_config.save_agent_index_dir())
        # persistence_manager = LocalStateManager(agent_config).load() # TODO: implement load
        # TODO: load prior agent state
        if persona and persona != agent_state.persona:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing persona {agent_state.persona} with {persona}", fg=typer.colors.YELLOW)
            agent_state.persona = persona
            # raise ValueError(f"Cannot override {agent_state.name} existing persona {agent_state.persona} with {persona}")
        if human and human != agent_state.human:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding existing human {agent_state.human} with {human}", fg=typer.colors.YELLOW)
            agent_state.human = human
            # raise ValueError(f"Cannot override {agent_config.name} existing human {agent_config.human} with {human}")

        # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
        if model and model != agent_state.llm_config.model:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model {agent_state.llm_config.model} with {model}", fg=typer.colors.YELLOW
            )
            agent_state.llm_config.model = model
        if context_window is not None and int(context_window) != agent_state.llm_config.context_window:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing context window {agent_state.llm_config.context_window} with {context_window}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.context_window = context_window
        if model_wrapper and model_wrapper != agent_state.llm_config.model_wrapper:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {agent_state.llm_config.model_wrapper} with {model_wrapper}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint != agent_state.llm_config.model_endpoint:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {agent_state.llm_config.model_endpoint} with {model_endpoint}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type != agent_state.llm_config.model_endpoint_type:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {agent_state.llm_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint_type = model_endpoint_type

        # Update the agent with any overrides
        ms.update_agent(agent_state)

        # create agent
        memgpt_agent = Agent(agent_state, interface=interface())

    else:  # create new agent
        # create new agent config: override defaults with args if provided
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)

        if agent is None:
            # determine agent name
            # agent_count = len(ms.list_agents(user_id=user.id))
            # agent = f"agent_{agent_count}"
            agent = utils.create_random_username()

        llm_config = MemGPTConfig.default_llm_config
        embedding_config = MemGPTConfig.default_embedding_config  # TODO allow overriding embedding params via CLI run

        # Allow overriding model specifics (model, model wrapper, model endpoint IP + type, context_window)
        if model and model != llm_config.model:
            typer.secho(f"{CLI_WARNING_PREFIX}Overriding default model {llm_config.model} with {model}", fg=typer.colors.YELLOW)
            llm_config.model = model
        if context_window is not None and int(context_window) != llm_config.context_window:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding default context window {llm_config.context_window} with {context_window}",
                fg=typer.colors.YELLOW,
            )
            llm_config.context_window = context_window
        if model_wrapper and model_wrapper != llm_config.model_wrapper:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {llm_config.model_wrapper} with {model_wrapper}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint != llm_config.model_endpoint:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {llm_config.model_endpoint} with {model_endpoint}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type != llm_config.model_endpoint_type:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {llm_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            llm_config.model_endpoint_type = model_endpoint_type

        agent_state = AgentState(
            name=agent,
            user_id=user.id,
            persona=persona if persona else MemGPTConfig.persona,
            human=human if human else MemGPTConfig.human,
            preset=preset if preset else MemGPTConfig.preset,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )
        ms.create_agent(agent_state)

        typer.secho(f"->  🤖 Using persona profile '{agent_state.persona}'", fg=typer.colors.WHITE)
        typer.secho(f"->  🧑 Using human profile '{agent_state.human}'", fg=typer.colors.WHITE)

        # Supress llama-index noise
        # TODO(swooders) add persistence manager code? or comment out?
        # with suppress_stdout():
        # TODO: allow configrable state manager (only local is supported right now)
        # persistence_manager = LocalStateManager(agent_config)  # TODO: insert dataset/pre-fill

        # create agent
        try:
            memgpt_agent = presets.create_agent_from_preset(
                agent_state=agent_state,
                interface=interface(),
            )
            save_agent(agent=memgpt_agent, ms=ms)
        except ValueError as e:
            # TODO(swooders) what's the equivalent cleanup code for the new DB refactor?
            typer.secho(f"Failed to create agent from provided information:\n{e}", fg=typer.colors.RED)
            # # Delete the directory of the failed agent
            # try:
            #     # Path to the specific file
            #     agent_config_file = agent_config.agent_config_path

            #     # Check if the file exists
            #     if os.path.isfile(agent_config_file):
            #         # Delete the file
            #         os.remove(agent_config_file)

            #     # Now, delete the directory along with any remaining files in it
            #     agent_save_dir = os.path.join(MEMGPT_DIR, "agents", agent_config.name)
            #     shutil.rmtree(agent_save_dir)
            # except:
            #     typer.secho(f"Failed to delete agent directory during cleanup:\n{e}", fg=typer.colors.RED)
            sys.exit(1)
        typer.secho(f"🎉 Created new agent '{agent_state.name}' (id={agent_state.id})", fg=typer.colors.GREEN)

    # pretty print agent config
    # printd(json.dumps(vars(agent_config), indent=4, sort_keys=True, ensure_ascii=JSON_ENSURE_ASCII))
    # printd(json.dumps(agent_init_state), indent=4, sort_keys=True, ensure_ascii=JSON_ENSURE_ASCII))

    # configure llama index
    original_stdout = sys.stdout  # unfortunate hack required to suppress confusing print statements from llama index
    sys.stdout = io.StringIO()
    embed_model = embedding_model(config=agent_state.embedding_config, user_id=user.id)
    service_context = ServiceContext.from_defaults(
        llm=None, embed_model=embed_model, chunk_size=agent_state.embedding_config.embedding_chunk_size
    )
    set_global_service_context(service_context)
    sys.stdout = original_stdout

    # start event loop
    from memgpt.main import run_agent_loop

    print()  # extra space
    run_agent_loop(memgpt_agent, first, ms, no_verify)  # TODO: add back no_verify


def delete_agent(
    agent_name: Annotated[str, typer.Option(help="Specify agent to delete")],
    user_id: Annotated[Optional[str], typer.Option(help="User ID to associate with the agent.")] = None,
):
    """Delete an agent from the database"""
    # use client ID is no user_id provided
    ms = MetadataStore()
    if user_id is None:
        user = create_default_user_or_exit(ms)
    else:
        user = ms.get_user(user_id=uuid.UUID(user_id))

    try:
        agent = ms.get_agent(agent_name=agent_name, user_id=user.id)
    except Exception as e:
        typer.secho(f"Failed to get agent {agent_name}\n{e}", fg=typer.colors.RED)
        sys.exit(1)

    if agent is None:
        typer.secho(f"Couldn't find agent named '{agent_name}' to delete", fg=typer.colors.RED)
        sys.exit(1)

    confirm = questionary.confirm(f"Are you sure you want to delete agent '{agent_name}' (id={agent.id})?", default=False).ask()
    if confirm is None:
        raise KeyboardInterrupt
    if not confirm:
        typer.secho(f"Cancelled agent deletion '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
        return

    try:
        ms.delete_agent(agent_id=agent.id)
        typer.secho(f"🕊️ Successfully deleted agent '{agent_name}' (id={agent.id})", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to delete agent '{agent_name}' (id={agent.id})", fg=typer.colors.RED)
        sys.exit(1)


def attach(
    agent_name: Annotated[str, typer.Option(help="Specify agent to attach data to")],
    data_source: Annotated[str, typer.Option(help="Data source to attach to agent")],
    user_id: uuid.UUID = None,
):
    # use client ID is no user_id provided
    if user_id is None:
        user_id = uuid.UUID(MemGPTConfig.anon_clientid)
    try:
        # loads the data contained in data source into the agent's memory
        from memgpt.agent_store.storage import StorageConnector, TableType
        from tqdm import tqdm

        ms = MetadataStore()
        agent = ms.get_agent(agent_name=agent_name, user_id=user_id)
        assert agent is not None, f"No agent found under agent_name={agent_name}, user_id={user_id}"
        source = ms.get_source(source_name=data_source, user_id=user_id)
        assert source is not None, f"Source {data_source} does not exist for user {user_id}"

        # get storage connectors
        with suppress_stdout():
            source_storage = StorageConnector.get_storage_connector(TableType.PASSAGES, user_id=user_id)
            dest_storage = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, user_id=user_id, agent_id=agent.id)

        size = source_storage.size({"data_source": data_source})
        typer.secho(f"Ingesting {size} passages into {agent.name}", fg=typer.colors.GREEN)
        page_size = 100
        generator = source_storage.get_all_paginated(filters={"data_source": data_source}, page_size=page_size)  # yields List[Passage]
        all_passages = []
        for i in tqdm(range(0, size, page_size)):
            passages = next(generator)

            # need to associated passage with agent (for filtering)
            for passage in passages:
                assert isinstance(passage, Passage), f"Generate yielded bad non-Passage type: {type(passage)}"
                passage.agent_id = agent.id

            # insert into agent archival memory
            dest_storage.insert_many(passages)
            all_passages += passages

        assert size == len(all_passages), f"Expected {size} passages, but only got {len(all_passages)}"

        # save destination storage
        dest_storage.save()

        # attach to agent
        source = ms.get_source(source_name=data_source, user_id=user_id)
        assert source is not None, f"source does not exist for source_name={data_source}, user_id={user_id}"
        source_id = source.id
        ms.attach_source(agent_id=agent.id, source_id=source_id, user_id=user_id)

        total_agent_passages = dest_storage.size()

        typer.secho(
            f"Attached data source {data_source} to agent {agent_name}, consisting of {len(all_passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
            fg=typer.colors.GREEN,
        )
    except KeyboardInterrupt:
        typer.secho("Operation interrupted by KeyboardInterrupt.", fg=typer.colors.YELLOW)


def version():
    import memgpt

    print(memgpt.__version__)
    return memgpt.__version__
