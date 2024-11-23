from collections.abc import Iterable
import json
from mitmproxy import http
import time
from jet.transformers import make_serializable
from jet.logger import logger

# Dictionary to store start times for requests
start_times = {}
chunks = []


def interceptor_callback(data: bytes) -> bytes | Iterable[bytes]:
    """
    This function will be called for each chunk of request/response body data that arrives at the proxy,
    and once at the end of the message with an empty bytes argument (b"").
    """
    global started_start_time  # Declare it global to modify the global variable

    decoded_data = data.decode('utf-8')
    chunk_dict = {}

    if not chunks:
        logger.log("Stream started")
        # Store the start time for the stream
        start_times["stream"] = time.time()
    try:
        chunk_dict = json.loads(decoded_data)
        if "message" in chunk_dict and chunk_dict["message"]["role"] == "assistant":
            content = chunk_dict["message"]["content"]
            logger.success(content, flush=True)
    except json.JSONDecodeError:
        pass

    chunks.append(decoded_data)
    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"
    logger.info(f"URL: {url}")
    # Log the serialized data as a JSON string
    request_dict = make_serializable(flow.request.data)
    logger.log(f"REQUEST:", request_dict, colors=["GRAY", "DEBUG"])
    logger.log(f"REQUEST KEYS:", list(
        request_dict.keys()), colors=["GRAY", "DEBUG"])
    start_times[flow.id] = time.time()  # Store the start time for the request


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    # Log the serialized data as a JSON string
    response_dict = make_serializable(flow.response.data)
    logger.log(f"RESPONSE:", response_dict, colors=["GRAY", "DEBUG"])
    logger.log(f"RESPONSE KEYS:", list(
        response_dict.keys()), colors=["GRAY", "DEBUG"])

    # Log combined chunks content
    # contents = []
    # for chunk in chunks:
    #     chunk_dict = json.loads(chunk)
    #     contents.append(chunk_dict['response'])
    logger.log("CHUNKS:", len(chunks), colors=["GRAY", "SUCCESS"])
    # content = "".join(contents)
    # logger.log("CONTENT:", content, colors=["GRAY", "SUCCESS"])

    end_time = time.time()  # Record the end time
    if "stream" in start_times:
        end_time = time.time()
        time_taken = end_time - start_times["stream"]
        logger.log("\n\nStream took:", f"{time_taken:.2f} seconds", colors=[
            "LOG",
            "BRIGHT_SUCCESS",
        ])
        started_start_time = None  # Clean up to avoid memory issues

    if flow.id in start_times:
        time_taken = end_time - start_times[flow.id]
        logger.log("Request total time took:", f"{time_taken:.2f} seconds", colors=[
            "LOG",
            "BRIGHT_SUCCESS",
        ])
        del start_times[flow.id]  # Clean up to avoid memory issues
    else:
        logger.warning(f"Start time for {flow.id} not found!")


def responseheaders(flow):
    """
    Set the response interceptor callback for streaming.
    """
    flow.response.stream = interceptor_callback


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://jetairm1:11435 -p 11434
