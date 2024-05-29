import uuid
import json

from .utils import get_local_time
from .constants import (
    JSON_ENSURE_ASCII,
)


def get_heartbeat(reason="Automated timer", include_location=False, location_name="San Francisco, CA, USA"):
    # Package the message with time and location
    formatted_time = get_local_time()
    packaged_message = {
        "type": "heartbeat",
        "reason": reason,
        "time": formatted_time,
    }

    if include_location:
        packaged_message["location"] = location_name

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def package_user_message(user_message, time=None, include_location=False, location_name="San Francisco, CA, USA", name=None):
    # Package the message with time and location
    formatted_time = time if time else get_local_time()
    packaged_message = {
        "type": "user_message",
        "message": user_message,
        "time": formatted_time,
    }

    if include_location:
        packaged_message["location"] = location_name

    if name:
        packaged_message["name"] = name

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)


def package_function_response(was_success, response_string, timestamp=None):
    formatted_time = get_local_time() if timestamp is None else timestamp
    packaged_message = {
        "status": "OK" if was_success else "Failed",
        "message": response_string,
        "time": formatted_time,
    }

    return json.dumps(packaged_message, ensure_ascii=JSON_ENSURE_ASCII)
