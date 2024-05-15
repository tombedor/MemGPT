from abc import ABC, abstractmethod
import json
import re

from colorama import Fore, Style, init

from memgpt.utils import printd
from memgpt.constants import CLI_WARNING_PREFIX, JSON_LOADS_STRICT

init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal


class AgentInterface(ABC):
    """Interfaces handle MemGPT-related events (observer pattern)"""

    @abstractmethod
    def user_message(self, msg):
        """MemGPT receives a user message"""
        raise NotImplementedError

    @abstractmethod
    def internal_monologue(self, msg):
        """MemGPT generates some internal monologue"""
        raise NotImplementedError

    @abstractmethod
    def assistant_message(self, msg):
        """MemGPT uses send_message"""
        raise NotImplementedError

    @abstractmethod
    def function_message(self, msg):
        """MemGPT calls a function"""
        raise NotImplementedError


class CLIInterface(AgentInterface):
    """Basic interface for dumping agent events to the command-line"""

    def internal_monologue(self, msg):
        # ANSI escape code for italic is '\x1B[3m'
        fstr = f"\x1B[3m{Fore.LIGHTBLACK_EX}💭 {{msg}}{Style.RESET_ALL}"
        print(fstr.format(msg=msg))

    def assistant_message(self, msg):
        fstr = f"{Fore.YELLOW}{Style.BRIGHT}🤖 {Fore.YELLOW}{{msg}}{Style.RESET_ALL}"
        print(fstr.format(msg=msg))

    def user_message(self, msg, raw=False, dump=False, debug=DEBUG):
        def print_user_message(icon, msg, printf=print):
            printf(f"{Fore.GREEN}{Style.BRIGHT}{icon} {Fore.GREEN}{msg}{Style.RESET_ALL}")

        def printd_user_message(icon, msg):
            return print_user_message(icon, msg)

        if not (raw or dump or debug):
            # we do not want to repeat the message in normal use
            return

        if raw:
            printd_user_message("🧑", msg)
            return
        else:
            try:
                msg_json = json.loads(msg, strict=JSON_LOADS_STRICT)
            except:
                printd(f"{CLI_WARNING_PREFIX}failed to parse user message into json")
                printd_user_message("🧑", msg)
                return
        if msg_json["type"] == "user_message":
            if dump:
                print_user_message("🧑", msg_json["message"])
                return
            msg_json.pop("type")
            printd_user_message("🧑", msg_json)
        elif msg_json["type"] == "heartbeat":
            if debug:
                msg_json.pop("type")
                printd_user_message("💓", msg_json)
            elif dump:
                print_user_message("💓", msg_json)
                return

        elif msg_json["type"] == "system_message":
            msg_json.pop("type")
            printd_user_message("🖥️", msg_json)
        else:
            printd_user_message("🧑", msg_json)

    def function_message(self, msg, debug=DEBUG):
        def print_function_message(icon, msg, color=Fore.RED, printf=print):
            printf(f"{color}{Style.BRIGHT}⚡{icon} [function] {color}{msg}{Style.RESET_ALL}")

        def printd_function_message(icon, msg, color=Fore.RED):
            return print_function_message(icon, msg, color, printf=(print if debug else printd))

        if isinstance(msg, dict):
            printd_function_message("", msg)
            return

        if msg.startswith("Success"):
            printd_function_message("🟢", msg)
        elif msg.startswith("Error: "):
            printd_function_message("🔴", msg)
        elif msg.startswith("Running "):
            if debug:
                printd_function_message("", msg)
            else:
                match = re.search(r"Running (\w+)\((.*)\)", msg)
                if match:
                    function_name = match.group(1)
                    function_args = match.group(2)
                    if function_name in ["archival_memory_insert", "archival_memory_search", "core_memory_replace", "core_memory_append"]:
                        if function_name in ["archival_memory_insert", "core_memory_append", "core_memory_replace"]:
                            print_function_message("🧠", f"updating memory with {function_name}")
                        elif function_name == "archival_memory_search":
                            print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            if function_name == "archival_memory_search":
                                output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                                print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                            elif function_name == "archival_memory_insert":
                                output = f'\t→ {msg_dict["content"]}'
                                print(f"{Style.BRIGHT}{Fore.RED}{output}{Style.RESET_ALL}")
                            else:
                                print(
                                    f'{Style.BRIGHT}\t{Fore.RED} {msg_dict["old_content"]}\n\t{Fore.GREEN}→ {msg_dict["new_content"]}{Style.RESET_ALL}'
                                )
                        except Exception as e:
                            printd(str(e))
                    elif function_name in ["conversation_search", "conversation_search_date"]:
                        print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                            print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                        except Exception as e:
                            printd(str(e))
                else:
                    printd(f"{CLI_WARNING_PREFIX}did not recognize function message")
                    printd_function_message("", msg)
        else:
            try:
                msg_dict = json.loads(msg, strict=JSON_LOADS_STRICT)
                if "status" in msg_dict and msg_dict["status"] == "OK":
                    printd_function_message("", str(msg), color=Fore.GREEN)
                else:
                    printd_function_message("", str(msg), color=Fore.RED)
            except Exception:
                print(f"{CLI_WARNING_PREFIX}did not recognize function message {type(msg)} {msg}")
                printd_function_message("", msg)

    @staticmethod
    def step_yield():
        pass
