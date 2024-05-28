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
