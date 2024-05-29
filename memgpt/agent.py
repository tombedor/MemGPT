import logging
import uuid
import inspect
import traceback
from typing import List, cast, Union

from attr import dataclass

from memgpt import system
from memgpt import constants
from memgpt.config import MemGPTConfig
from memgpt.metadata import MetadataStore
from memgpt.data_types import AgentState, Message
from memgpt.models import chat_completion_response
from memgpt.persistence_manager import PersistenceManager
from memgpt.system import package_function_response
from memgpt.llm_api_tools import create, is_context_overflow_error
from memgpt.utils import (
    get_tool_call_id,
    parse_json,
    printd,
    validate_function_response,
)
from memgpt.constants import (
    WARNING_PREFIX,
    SYSTEM,
)
from .functions.functions import ALL_FUNCTIONS

MAX_CHAINING_STEPS = 20


def initialize_message_sequence():
    naive_system_message = "\n".join(
        [
            SYSTEM,
            "\n",
            "<persona>",
            "I am a personal assistant. I am here to honestly converse with my user, and be helpful. My tone should be casual and conversational."
            "</persona>",
            "<human>",
            "An unknown human. They have not yet shared their name.",
            "</human>",
        ]
    )

    messages = [
        {"role": "system", "content": naive_system_message},
    ]

    return messages


class Agent(object):
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

        self.functions = [fs["json_schema"] for fs in ALL_FUNCTIONS.values()]
        self.functions_python = {k: v["python_function"] for k, v in ALL_FUNCTIONS.items()}

        self.persistence_manager = PersistenceManager(agent_state=self.agent_state)

        self._messages: List[Message] = []

        # Once the memory object is initialized, use it to "bake" the system message
        if "messages" in self.agent_state.state and self.agent_state.state["messages"] is not None:
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
            init_messages = initialize_message_sequence()
            init_messages_objs = []
            for msg in init_messages:
                init_messages_objs.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id, user_id=self.agent_state.user_id, model=MemGPTConfig.model, openai_message_dict=msg
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

    def _append_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.persist_messages(added_messages)

        new_messages = self._messages + added_messages  # append

        self._messages = new_messages

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
                    model=MemGPTConfig.model,
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
                        model=MemGPTConfig.model,
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
                        model=MemGPTConfig.model,
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
                    model=MemGPTConfig.model,
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
                    model=MemGPTConfig.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            heartbeat_request = False
            function_failed = False

        return self.AiResponse(messages=messages, heartbeat_request=heartbeat_request, function_failed=function_failed)

    @dataclass
    class AgentStepResult:
        heartbeat_request: bool
        function_failed: bool

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
                            openai_message_dict=packed_user_message,  # type: ignore
                        )
                    ] + ai_response.messages
            else:
                all_new_messages = ai_response.messages

            self._append_to_messages(all_new_messages)

            return self.AgentStepResult(
                heartbeat_request=ai_response.heartbeat_request,
                function_failed=ai_response.function_failed,
            )

        except Exception as e:
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            if is_context_overflow_error(e):
                raise ValueError(f"Context overflow error: {e}")
            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def step_chain(self, input_message: Union[str, Message]) -> List[Message]:
        next_input_message = input_message
        counter = 0
        while True:
            step_response = self.step(next_input_message)
            counter += 1

            # Chain stops
            if counter > MAX_CHAINING_STEPS:
                logging.debug(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif step_response.function_failed:
                next_input_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                continue  # always chain
            elif step_response.heartbeat_request:
                next_input_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                continue  # always chain
            # MemGPT no-op / yield
            else:
                break
        save_agent(self)
        return self._messages

    def update_state(self) -> AgentState:
        updated_state = {
            "functions": self.functions,
            "messages": [str(msg.id) for msg in self._messages],
        }

        self.agent_state = AgentState(
            user_id=self.agent_state.user_id,
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
