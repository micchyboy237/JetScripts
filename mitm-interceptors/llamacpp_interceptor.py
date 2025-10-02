import os
import shutil
import time
import traceback

from collections.abc import Iterable
from jet.transformers.object import make_serializable
from mitmproxy import http

from jet.utils.class_utils import get_class_name
from jet.logger import logger


LOGS_DIR = os.path.expanduser("~/.cache/logs/llamacpp-logs")


def generate_log_file_path():
    # Generate a timestamp and unique log file name
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_file_name = f"{timestamp}_{int(time.time())}.md"
    log_file_name = f"{int(time.time())}.md"
    log_file_path = os.path.realpath(os.path.join(
        LOGS_DIR, log_file_name).replace(' ', '_'))

    return log_file_path


def remove_old_files_by_limit(limit: str = 15):
    """
    Removes the oldest files or directories in `base_dir` to maintain only `limit` most recent items.
    """
    if not os.path.exists(LOGS_DIR):
        return

    existing_logs = sorted(
        (os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR)),
        key=os.path.getctime
    )

    while len(existing_logs) > limit:
        oldest = existing_logs.pop(0)
        if os.path.isdir(oldest):
            shutil.rmtree(oldest)  # Remove directory and contents
        else:
            os.remove(oldest)  # Remove file


def interceptor_callback(data: bytes) -> bytes | Iterable[bytes]:
    """
    This function will be called for each chunk of request/response body data that arrives at the proxy,
    and once at the end of the message with an empty bytes argument (b"").
    """

    if not data:  # Handle empty data
        return b""

    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """

    limit = 15

    log_file_path = generate_log_file_path()
    logger.basicConfig(filename=log_file_path)

    logger.log("\n")
    url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

    logger.newline()
    logger.log("request client_conn.id:", flow.client_conn.id,
               colors=["WHITE", "PURPLE"])

    logger.info(f"URL: {url}")

    request_dict = make_serializable(flow.request.data)

    logger.newline()
    logger.log("REQUEST KEYS:")
    for k, v in request_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
        else:
            logger.log(f"  {k}: <{type(v).__name__}>",
                        colors=["GRAY", "TEAL"])
    logger.log("REQUEST HEADERS:")
    headers = request_dict["headers"]
    if isinstance(headers, dict) and "fields" in headers:
        for k, v in headers["fields"]:
            logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
    else:
        logger.log(f"  {headers}", colors=["GRAY", "TEAL"])

    if limit:
        remove_old_files_by_limit(limit)

    


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """

    logger.newline()
    logger.log("response client_conn.id:",
               flow.client_conn.id, colors=["WHITE", "PURPLE"])

    response_dict = make_serializable(flow.response.data)

    logger.newline()
    logger.log("RESPONSE KEYS:")
    for k, v in response_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
        else:
            logger.log(f"  {k}: <{type(v).__name__}>",
                        colors=["GRAY", "TEAL"])
    logger.log("RESPONSE HEADERS:")
    headers = response_dict["headers"]
    if isinstance(headers, dict) and "fields" in headers:
        for k, v in headers["fields"]:
            logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
    else:
        logger.log(f"  {headers}", colors=["GRAY", "TEAL"])

    
def responseheaders(flow):
    """
    Set the response interceptor callback for streaming.
    """
    flow.response.stream = interceptor_callback


def error(flow: http.HTTPFlow):
    """Kills the flow if it has an error different to HTTPSyntaxException.
    Sometimes, web scanners generate malformed HTTP syntax on purpose and we do not want to kill these requests.
    """
    # from mitmproxy.exceptions import HttpSyntaxException
    # if flow.error is not None and not isinstance(flow.error, HttpSyntaxException):
    #     flow.kill()
    logger.newline()
    logger.error("Error occurred in mitmproxy:")

    if flow.error is not None:
        logger.warning(f"Error type: {get_class_name(flow.error)}")
        logger.error(flow.error)

    # Log the full stack trace
    logger.error("Stack trace:")
    logger.error(traceback.format_exc())  # This captures the stack trace

    # Log the full stack trace
    # logger.warning("Stack trace:")
    # logger.error(traceback.format_exc())  # This captures the stack trace
