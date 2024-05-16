import uuid
import inspect
import traceback
from typing import List, cast, Union

from attr import dataclass

from memgpt.metadata import MetadataStore
from memgpt.data_types import AgentState, Message, LLMConfig, EmbeddingConfig, Preset
from memgpt.models import chat_completion_response
from memgpt.persistence_manager import PersistenceManager
from memgpt.system import get_login_event, package_function_response, package_summarize_message, get_initial_boot_messages
from memgpt.memory import InContextMemory, summarize_messages
from memgpt.llm_api_tools import create, is_context_overflow_error
from memgpt.utils import (
    get_tool_call_id,
    parse_json,
    printd,
    count_tokens,
    validate_function_response,
)
from memgpt.constants import (
    DEFAULT_PERSONA,
    MESSAGE_SUMMARY_WARNING_FRAC,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    LLM_MAX_TOKENS,
    WARNING_PREFIX,
    SYSTEM,
)
from .errors import LLMError
from .functions.functions import load_all_function_sets


def initialize_memory(ai_notes: str, human_notes: str):
    assert ai_notes
    assert human_notes
    memory = InContextMemory(human_char_limit=CORE_MEMORY_HUMAN_CHAR_LIMIT, persona_char_limit=CORE_MEMORY_PERSONA_CHAR_LIMIT)
    memory.edit_persona(ai_notes)
    memory.edit_human(human_notes)
    return memory


# TODO (tbedor) this should be generated dynamically
def construct_system_with_memory(
    memory: InContextMemory,
):

    human = memory.human
    assert type(human) == str
    full_system_message = "\n".join(
        [
            SYSTEM,
            "\n",
            "<persona>",
            DEFAULT_PERSONA,
            "</persona>",
            "<human>",
            human,
            "</human>",
        ]
    )
    return full_system_message


def initialize_message_sequence(
    memory: InContextMemory,
):

    full_system_message = construct_system_with_memory(memory)
    first_user_message = get_login_event()  # event letting MemGPT know the user just logged in

    initial_boot_messages = get_initial_boot_messages()
    messages = (
        [
            {"role": "system", "content": full_system_message},
        ]
        + initial_boot_messages
        + [
            {"role": "user", "content": first_user_message},
        ]
    )

    return messages


class Agent(object):
    @classmethod
    def initialize_with_configs(
        cls,
        preset: Preset,
        created_by: uuid.UUID,
        name: str,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
    ):

        assert created_by is not None, "Must provide created_by field when creating an Agent from a Preset"
        assert llm_config is not None, "Must provide llm_config field when creating an Agent from a Preset"
        assert embedding_config is not None, "Must provide embedding_config field when creating an Agent from a Preset"
        assert name is not None, "must proide agent name"

        # if agent_state is also provided, override any preset values
        init_agent_state = AgentState(
            name=name,
            user_id=created_by,
            human=preset.human,
            llm_config=llm_config,
            embedding_config=embedding_config,
            state={
                "persona": DEFAULT_PERSONA,
                "human": preset.human,
                "system": SYSTEM,
                "messages": None,
            },
        )
        return cls(agent_state=init_agent_state)

    def __init__(
        self,
        agent_state: AgentState,
    ):

        # An agent can also be created directly from AgentState
        assert agent_state.state is not None and agent_state.state != {}, "AgentState.state cannot be empty"

        # Assume the agent_state passed in is formatted correctly
        init_agent_state = agent_state

        # Hold a copy of the state that was used to init the agent
        self.agent_state = init_agent_state

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model

        all_functions = load_all_function_sets()
        self.functions = [fs["json_schema"] for fs in all_functions.values()]
        self.functions_python = {k: v["python_function"] for k, v in all_functions.items()}

        assert all([callable(f) for k, f in self.functions_python.items()]), self.functions_python

        if "human" not in self.agent_state.state:
            raise ValueError(f"'human' not found in provided AgentState")
        self.memory = initialize_memory(ai_notes=self.agent_state.state["persona"], human_notes=self.agent_state.state["human"])

        self.persistence_manager = PersistenceManager(agent_state=self.agent_state)

        # Controls if the convo memory pressure warning is triggered
        # When an alert is sent in the message queue, set this to True (to avoid repeat alerts)
        # When the summarizer is run, set this back to False (to reset)
        self.agent_alerted_about_memory_pressure = False

        self._messages: List[Message] = []

        # Once the memory object is initialized, use it to "bake" the system message
        if "messages" in self.agent_state.state and self.agent_state.state["messages"] is not None:
            # print(f"Agent.__init__ :: loading, state={agent_state.state['messages']}")
            if not isinstance(self.agent_state.state["messages"], list):
                raise ValueError(f"'messages' in AgentState was bad type: {type(self.agent_state.state['messages'])}")
            assert all([isinstance(msg, str) for msg in self.agent_state.state["messages"]])

            # Convert to IDs, and pull from the database
            raw_messages = [
                self.persistence_manager.recall_memory.storage.get(id=uuid.UUID(msg_id)) for msg_id in self.agent_state.state["messages"]
            ]
            assert all([isinstance(msg, Message) for msg in raw_messages]), (raw_messages, self.agent_state.state["messages"])
            self._messages.extend([cast(Message, msg) for msg in raw_messages if msg is not None])

        else:
            init_messages = initialize_message_sequence(
                self.memory,
            )
            init_messages_objs = []
            for msg in init_messages:
                init_messages_objs.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id, user_id=self.agent_state.user_id, model=self.model, openai_message_dict=msg
                    )
                )
            assert all([isinstance(msg, Message) for msg in init_messages_objs]), (init_messages_objs, init_messages)
            self._append_to_messages(added_messages=[cast(Message, msg) for msg in init_messages_objs if msg is not None])

        # Create the agent in the DB
        self.update_state()

    @property
    def messages(self) -> List[dict]:
        """Getter method that converts the internal Message list into OpenAI-style dicts"""
        return [msg.to_openai_dict() for msg in self._messages]

    @messages.setter
    def messages(self, value):
        raise Exception("Modifying message list directly not allowed")

    def _trim_messages(self, num):
        """Trim messages from the front, not including the system message"""

        new_messages = [self._messages[0]] + self._messages[num:]
        self._messages = new_messages

    def _prepend_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.prepend to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.persist_messages(added_messages)

        new_messages = [self._messages[0]] + added_messages + self._messages[1:]  # prepend (no system)
        self._messages = new_messages

    def _append_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.persist_messages(added_messages)

        new_messages = self._messages + added_messages  # append

        self._messages = new_messages

    def append_to_messages(self, added_messages: List[dict]):
        """An external-facing message append, where dict-like messages are first converted to Message objects"""
        added_messages_objs = [
            Message.dict_to_message(
                agent_id=self.agent_state.id,
                user_id=self.agent_state.user_id,
                model=self.model,
                openai_message_dict=msg,
            )
            for msg in added_messages
        ]
        self._append_to_messages(added_messages_objs)

    def _get_ai_reply(
        self,
        message_sequence: List[dict],
        function_call: str = "auto",
    ) -> chat_completion_response.ChatCompletionResponse:
        """Get response from LLM API"""
        try:
            response = create(
                agent_state=self.agent_state,
                messages=message_sequence,
                functions=self.functions,
                function_call=function_call,
            )
            # special case for 'length'
            if response.choices[0].finish_reason == "length":
                raise Exception("Finish reason was length (maximum context length)")

            # catches for soft errors
            if response.choices[0].finish_reason not in ["stop", "function_call", "tool_calls"]:
                raise Exception(f"API call finish with bad finish reason: {response}")

            # unpack with response.choices[0].message.content
            return response
        except Exception as e:
            raise e

    @dataclass
    class AiResponse:
        messages: List[Message]
        heartbeat_request: bool
        function_failed: bool

    def _handle_ai_response(self, response_message: chat_completion_response.Message) -> AiResponse:
        """Handles parsing and function execution"""

        messages = []  # append these to the history when done

        # Step 2: check if LLM wanted to call a function
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            if response_message.tool_calls is not None and len(response_message.tool_calls) > 1:
                # raise NotImplementedError(f">1 tool call not supported")
                # TODO eventually support sequential tool calling
                printd(f">1 tool call not supported, using index=0 only\n{response_message.tool_calls}")
                response_message.tool_calls = [response_message.tool_calls[0]]

            # generate UUID for tool call
            tool_call_id = get_tool_call_id()  # needs to be a string for JSON
            response_message.tool_calls[0].id = tool_call_id

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            printd(f"Function call message: {messages[-1]}")

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            # Failure case 1: function name is wrong
            function_call = response_message.tool_calls[0].function
            function_name = function_call.name
            printd(f"Request to call function {function_name} with tool_call_id: {tool_call_id}")
            try:
                function_to_call = self.functions_python[function_name]
            except KeyError:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                return self.AiResponse(
                    messages=messages, heartbeat_request=False, function_failed=True
                )  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = function_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                return self.AiResponse(
                    messages=messages, heartbeat_request=False, function_failed=True
                )  # force a heartbeat to allow agent to handle error

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)
            if not (isinstance(heartbeat_request, bool) or heartbeat_request is None):
                printd(
                    f"{WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = False

            # Failure case 3: function failed during execution
            try:
                spec = inspect.getfullargspec(function_to_call).annotations

                for name, arg in function_args.items():
                    if isinstance(function_args[name], dict):
                        function_args[name] = spec[name](**function_args[name])

                function_args["self"] = self  # need to attach self to arg since it's dynamically linked

                function_response = function_to_call(**function_args)
                if function_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
                    # with certain functions we rely on the paging mechanism to handle overflow
                    truncate = False
                else:
                    # but by default, we add a truncation safeguard to prevent bad functions from
                    # overflow the agent context window
                    truncate = True
                function_response_string = validate_function_response(function_response, truncate=truncate)
                function_args.pop("self", None)
                function_response = package_function_response(True, function_response_string)
                function_failed = False
            except Exception as e:
                function_args.pop("self", None)
                # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                error_msg = f"Error calling function {function_name}: {str(e)}"
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                printd(error_msg_user)
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                return self.AiResponse(
                    messages=messages, heartbeat_request=False, function_failed=True
                )  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        "tool_call_id": tool_call_id,
                    },
                )
            )  # extend conversation with function response

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            heartbeat_request = False
            function_failed = False

        return self.AiResponse(messages=messages, heartbeat_request=heartbeat_request, function_failed=function_failed)

    @dataclass
    class AgentStepResult:
        message_dicts: List[dict]
        heartbeat_request: bool
        function_failed: bool
        active_memory_warning: bool

    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
    ) -> AgentStepResult:
        """Top-level event message handler for the MemGPT agent"""

        try:
            # Step 0: add user message
            if user_message is not None:
                if isinstance(user_message, Message):
                    user_message_text = user_message.text
                elif isinstance(user_message, str):
                    user_message_text = user_message
                else:
                    raise ValueError(f"Bad type for user_message: {type(user_message)}")

                packed_user_message = {"role": "user", "content": user_message_text}

                input_message_sequence = self.messages + [packed_user_message]
            # Alternatively, the requestor can send an empty user message
            else:
                input_message_sequence = self.messages
                packed_user_message = None

            if len(input_message_sequence) > 1 and input_message_sequence[-1]["role"] != "user":
                printd(f"{WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            response = self._get_ai_reply(
                message_sequence=input_message_sequence,
            )

            # Step 2: check if LLM wanted to call a function
            # (if yes) Step 3: call the function
            # (if yes) Step 4: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message.copy()
            ai_response = self._handle_ai_response(response_message)

            # Step 4: extend the message history
            if user_message is not None:
                if isinstance(user_message, Message):
                    all_new_messages = [user_message] + ai_response.messages
                else:
                    all_new_messages = [
                        Message.dict_to_message(
                            agent_id=self.agent_state.id,
                            user_id=self.agent_state.user_id,
                            model=self.model,
                            openai_message_dict=packed_user_message,  # type: ignore
                        )
                    ] + ai_response.messages
            else:
                all_new_messages = ai_response.messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False
            # We can't do summarize logic properly if context_window is undefined
            if self.agent_state.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(f"{WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                print(f"{self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
                )
            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window):
                printd(
                    f"{WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this
            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )

            self._append_to_messages(all_new_messages)
            all_new_messages_dicts = [msg.to_openai_dict() for msg in all_new_messages]

            return self.AgentStepResult(
                message_dicts=all_new_messages_dicts,
                heartbeat_request=ai_response.heartbeat_request,
                function_failed=ai_response.function_failed,
                active_memory_warning=active_memory_warning,
            )

        except Exception as e:
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.step(user_message)
            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def get_system_message_text(self):
        return self._messages[0].text

    def set_system_message_text(self, new_system_message_text: str):
        system_message = self._messages[0]
        system_message.text = new_system_message_text
        system_message.id = uuid.uuid4()
        self.persistence_manager.persist_messages([system_message])
        self._messages[0] = system_message
        save_agent(self)

    def summarize_messages_inplace(self):
        assert self.messages[0]["role"] == "system", f"self.messages[0] should be system (instead got {self.messages[0]})"

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in self.messages]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)
        candidate_messages_to_summarize = self.messages[1:]
        token_counts = token_counts[1:]

        candidate_messages_to_summarize = candidate_messages_to_summarize[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
        token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]

        printd(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        printd(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        printd(f"token_counts={token_counts}")
        printd(f"message_buffer_token_count={message_buffer_token_count}")
        printd(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        printd(f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}")

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(self.messages)}, preserve_N={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}]"
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            printd(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if self.messages[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if self.messages[new_cutoff]["role"] == "user":
                    printd(f"Shifted cutoff {new_cutoff} is still a 'user', ignoring...")
                cutoff = new_cutoff
        except IndexError:
            pass

        # Make sure the cutoff isn't on a 'tool' or 'function'
        while self.messages[cutoff]["role"] in ["tool", "function"] and cutoff < len(self.messages):
            printd(f"Selected cutoff {cutoff} was a 'tool', shifting one...")
            cutoff += 1

        message_sequence_to_summarize = self.messages[1:cutoff]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            printd(f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self.messages)}")

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(f"{WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
            print(f"{self.agent_state}")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
            )
        summary = summarize_messages(agent_state=self.agent_state, message_sequence_to_summarize=message_sequence_to_summarize)
        printd(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(summary, summary_message_count)
        printd(f"Packaged into message: {summary_message}")

        prior_len = len(self.messages)
        self._trim_messages(cutoff)
        packed_summary_message = {"role": "user", "content": summary_message}
        self._prepend_to_messages(
            [
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                )
            ]
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False
        save_agent(self)

        printd(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def update_state(self) -> AgentState:
        updated_state = {
            "persona": self.memory.persona,
            "human": self.memory.human,
            "functions": self.functions,
            "messages": [str(msg.id) for msg in self._messages],
        }

        self.agent_state = AgentState(
            name=self.agent_state.name,
            user_id=self.agent_state.user_id,
            human=self.agent_state.human,
            llm_config=self.agent_state.llm_config,
            embedding_config=self.agent_state.embedding_config,
            id=self.agent_state.id,
            created_at=self.agent_state.created_at,
            state=updated_state,
        )
        return self.agent_state


def save_agent(agent: Agent):
    """Save agent to metadata store"""
    ms = MetadataStore()

    agent.update_state()
    agent_state = agent.agent_state

    if ms.get_agent(agent_id=agent_state.id):
        ms.update_agent(agent_state)
    else:
        ms.create_agent(agent_state)
