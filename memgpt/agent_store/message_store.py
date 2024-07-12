import os
from typing import Dict, List
from urllib.parse import urlparse
from uuid import UUID

from redis import StrictRedis
from redis_cache import RedisCache

from memgpt.constants import SESSION_MAKER
from memgpt.data_types import Message
from toolz import pipe

from typing import List

from memgpt.agent_store.storage import RecallMemoryModel


redis_url = urlparse(os.environ["REDIS_URL"])

if redis_url.password:
    redis_client = StrictRedis(
        host=redis_url.hostname, port=redis_url.port, db=1, ssl=True, ssl_cert_reqs=None, decode_responses=True, password=redis_url.password  # type: ignore
    )
else:
    redis_client = StrictRedis(host=redis_url.hostname, port=redis_url.port, db=1, ssl=False, decode_responses=True)  # type: ignore
    
assert redis_client.ping()
cache = RedisCache(redis_client = redis_client)


def get_message(id: UUID) -> Message:
    @cache.cache()
    def get_message_json(id: UUID) -> Dict:
        with SESSION_MAKER() as session:
            return pipe(
               session.query(RecallMemoryModel).filter(RecallMemoryModel.id == id).first(), # type: ignore 
               lambda x: x.to_record(),
               lambda x: x.to_openai_dict(),
            )
    return pipe(
        id,
        get_message_json,
        Message.dict_to_message,
    ) # type: ignore
    
    
def persist_messages(messages: List[Message]) -> None:
    [persist_message_if_not_exist(m) for m in messages]

def persist_message_if_not_exist(message: Message) -> None:
    with SESSION_MAKER() as session:
        if not session.query(Message).filter(Message.id == message.id).first(): # type: ignore
            session.add(message)
            session.commit()
