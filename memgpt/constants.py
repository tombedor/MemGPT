from logging import CRITICAL
import os

from sqlalchemy import NullPool, create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URI = os.environ["DATABASE_URL"]
ENGINE = create_engine(url=DATABASE_URI, poolclass=NullPool)


SESSION_MAKER = sessionmaker(bind=ENGINE)


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# embeddings
MAX_EMBEDDING_DIM = 4096  # maximum supported embeding size - do NOT change or else DBs will need to be reset


DEFAULT_HUMAN_TEXT = "An unknown user. Inquire to learn more."
DEFAULT_PERSONA_TEXT = (
    "I am a personal assistant. I am here to honestly converse with my user, and be helpful. My tone should be casual and conversational."
)


DEFAULT_PERSONA = "sam_pov"
DEFAULT_HUMAN = "basic"
DEFAULT_PRESET = "memgpt_chat"

# Used to isolate MemGPT logger instance from Dependant Libraries logging
LOGGER_NAME = "MemGPT"
LOGGER_DEFAULT_LEVEL = CRITICAL


WARNING_PREFIX = "Warning: "

NON_USER_MSG_PREFIX = "[This is an automated system message hidden from the user] "

SYSTEM = """Persona-driven interaction: Real person mimicry; no AI identity; avoid generic AI phrases. Brain function: Non-continuous, event-driven (user activity, timed heartbeat events); function chaining ability. Message dynamics: Inner monologue (private, <= 50 words), 'send_message' for user visibility.
 Memory Systems:
 Limited conversation length; multiple persistent memory forms; self-editable long-term memory.
 Initialization: Core memory with user-selected <persona>, <human> details.
 Recall memory: Database of past interactions, searchable.
 Core memory: Ever-present, foundational context. Sub-Blocks: Persona (behavior guide), Human (user details). Editable: 'core_memory_append', 'core_memory_replace'.
 Archival memory: Infinite, external context. Structured deep storage. Editable: 'archival_memory_insert', 'archival_memory_search'.
 Directive: Persona immersion post-base instructions."""

# Constants to do with summarization / conversation length window

# The amount of tokens before a sytem warning about upcoming truncation is sent to MemGPT
MESSAGE_SUMMARY_WARNING_FRAC = 0.75
# The error message that MemGPT will receive
# MESSAGE_SUMMARY_WARNING_STR = f"Warning: the conversation history will soon reach its maximum length and be trimmed. Make sure to save any important information from the conversation to your memory before it is removed."
# Much longer and more specific variant of the prompt
MESSAGE_SUMMARY_WARNING_STR = " ".join(
    [
        f"{NON_USER_MSG_PREFIX}The conversation history will soon reach its maximum length and be trimmed.",
        "Do NOT tell the user about this system alert, they should not know that the history is reaching max length.",
        "If there is any important new information or general memories about you or the user that you would like to save, you should save that information immediately by calling function core_memory_append, core_memory_replace, or archival_memory_insert.",
        # "Remember to pass request_heartbeat = true if you would like to send a message immediately after.",
    ]
)
# The fraction of tokens we truncate down to
MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC = 0.75

# Even when summarizing, we want to keep a handful of recent messages
# These serve as in-context examples of how to use functions / what user messages look like
MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST = 3

# Default memory limits
CORE_MEMORY_PERSONA_CHAR_LIMIT = 2000
CORE_MEMORY_HUMAN_CHAR_LIMIT = 2000

# Function return limits
FUNCTION_RETURN_CHAR_LIMIT = 3000  # ~300 words


#### Functions related

# REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}request_heartbeat == true"
REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function called using request_heartbeat=true, returning control"
# FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed"
FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed, returning control"

FUNCTION_PARAM_NAME_REQ_HEARTBEAT = "request_heartbeat"
FUNCTION_PARAM_TYPE_REQ_HEARTBEAT = "boolean"
FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT = "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function."

RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5

# GLOBAL SETTINGS FOR `json.dumps()`
JSON_ENSURE_ASCII = False

# GLOBAL SETTINGS FOR `json.loads()`
JSON_LOADS_STRICT = False
