from collections.abc import Iterable
from datetime import datetime
import json
import os
from mitmproxy import http
import time
from jet.transformers import make_serializable, format_prompt_log
from jet.logger import logger

file_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(file_dir, "jet-logs")
os.makedirs(log_dir, exist_ok=True)

# Generate a more readable timestamp and unique log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"{timestamp}_{
                             int(time.time())}.log")

logger.log("Log File Path:", log_file_path, colors=["GRAY", "INFO"])

# Dictionary to store start times for requests
start_times: dict[str, float] = {}
chunks: list[str] = []


def generate_log_entry(flow: http.HTTPFlow) -> str:
    """
    Generates a formatted log entry with metadata, prompt, and response.
    """
    global chunks

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    url = f"{flow.request.scheme}://{flow.request.host}{flow.request.path}"

    # Get last user prompt
    prompt_log = flow.request.data.content.decode('utf-8')
    prompt_log = json.loads(prompt_log)
    prompt_log = json.dumps(prompt_log, indent=2)
    # prompt = flow.request.data.content
    # logger.debug(f"PROMPT TYPE: {type(prompt)}")
    # logger.debug(f"PROMPT CONTENT:\n{prompt}")
    # try:
    #     prompt_log = format_prompt_log(json.loads(prompt))
    #     logger.log("PROMPT LOG:")
    #     logger.debug(prompt_log)
    # except json.JSONDecodeError:
    #     prompt_log = prompt

    # Get last assistant response
    contents = []
    for chunk in chunks:
        try:
            contents.append(chunk)
        except json.JSONDecodeError:
            pass
    response = "".join(contents)

    log_entry = (
        f"\n{'-'*80}\n"
        f"Timestamp: {timestamp}\n"
        f"Flow ID: {flow.id}\n"
        f"URL: {url}\n"
        f"Prompt:\n{prompt_log}\n"
        f"Response:\n{response}\n"
        f"{'-'*80}\n"
    )
    return log_entry


def interceptor_callback(data: bytes) -> bytes | Iterable[bytes]:
    """
    This function will be called for each chunk of request/response body data that arrives at the proxy,
    and once at the end of the message with an empty bytes argument (b"").
    """
    global chunks

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
            chunks.append(content)
            logger.success(content, flush=True)
    except json.JSONDecodeError:
        pass

    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    logger.log("\n")
    url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"
    logger.info(f"URL: {url}")
    # Log the serialized data as a JSON string
    request_dict = make_serializable(flow.request.data)
    logger.log(f"REQUEST KEYS:", list(
        request_dict.keys()), colors=["GRAY", "INFO"])
    logger.log(f"REQUEST:")
    logger.debug(json.dumps(request_dict, indent=2))
    start_times[flow.id] = time.time()  # Store the start time for the request


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    global chunks

    logger.log("\n")
    # Log the serialized data as a JSON string
    response_dict = make_serializable(flow.response.data)
    logger.log(f"RESPONSE KEYS:", list(
        response_dict.keys()), colors=["GRAY", "INFO"])
    logger.log(f"RESPONSE:")
    logger.debug(json.dumps(response_dict, indent=2))

    end_time = time.time()  # Record the end time
    if "stream" in start_times:
        end_time = time.time()
        time_taken = end_time - start_times["stream"]
        logger.log("\n\nStream took:", f"{time_taken:.2f} seconds", colors=[
            "LOG",
            "BRIGHT_SUCCESS",
        ])

    if flow.id in start_times:
        time_taken = end_time - start_times[flow.id]
        logger.log("Request total time took:", f"{time_taken:.2f} seconds", colors=[
            "LOG",
            "BRIGHT_SUCCESS",
        ])
        del start_times[flow.id]  # Clean up to avoid memory issues
    else:
        logger.warning(f"Start time for {flow.id} not found!")

    if flow.request.path.startswith("/api/chat") or flow.request.path.startswith("/api/generate"):
        logger.log("FIRST 5 CHUNKS:")
        logger.debug(chunks[0:5])

        # Log prompt and response with metadata to the log file
        log_entry = generate_log_entry(flow)
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    chunks = []  # Clean up to avoid memory issues


def responseheaders(flow):
    """
    Set the response interceptor callback for streaming.
    """
    flow.response.stream = interceptor_callback


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://jetairm1:11435 -p 11434
