from collections.abc import Iterable
from datetime import datetime
import json
import os
from mitmproxy import http
import time
from jet.transformers import make_serializable, format_prompt_log
from jet.logger import logger

LOGS_DIR = "jet-logs"
log_file_path = None

# Dictionary to store start times for requests
start_times: dict[str, float] = {}
chunks: list[str] = []


def generate_log_file_path(logs_dir, base_dir=None, limit=10):
    # Determine the base directory
    if base_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))

    # Create the log directory if it doesn't exist
    log_dir = os.path.join(base_dir, logs_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Maintain only the `limit` most recent files
    existing_logs = sorted(
        (os.path.join(log_dir, f) for f in os.listdir(log_dir)
         if os.path.isfile(os.path.join(log_dir, f))),
        key=os.path.getctime
    )
    while len(existing_logs) >= limit:
        os.remove(existing_logs.pop(0))

    # Generate a timestamp and unique log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{timestamp}_{int(time.time())}.md"

    return os.path.join(log_dir, log_file_name)


def generate_log_entry(flow: http.HTTPFlow) -> str:
    """
    Generates a formatted log entry with metadata, prompt, and response.
    """
    global chunks

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    url = f"{flow.request.scheme}://{flow.request.host}{flow.request.path}"

    # Get last user prompt
    prompt_log = flow.request.data.content.decode('utf-8')
    prompt_log_dict = json.loads(prompt_log)
    model = prompt_log_dict['model']
    messages = prompt_log_dict['messages']
    prompt_msgs = []
    for item_idx, item in enumerate(messages):
        prompt_msg = (
            f"Message {item_idx + 1}.\n"
            f"- **Role**: {item['role']}\n"
            f"- **Content**: {item['content']}\n"
        ).strip()
        prompt_msgs.append(prompt_msg)

    prompt_log = "\n\n".join(prompt_msgs)

    # Get last assistant response
    contents = []
    for chunk in chunks:
        try:
            contents.append(chunk)
        except json.JSONDecodeError:
            pass
    response = "".join(contents)

    log_entry = (
        f"## Request Info\n\n"
        f"- **Timestamp**: {timestamp}\n"
        f"- **Flow ID**: {flow.id}\n"
        f"- **URL**: {url}\n"
        f"- **Model**: {model}\n"
        f"## Prompt ({len(messages)})\n\n```markdown\n{prompt_log}\n```\n"
        f"## Response\n\n```markdown\n{response}\n```\n"
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
        # logger.log("Stream started")
        # Store the start time for the stream
        # start_times["stream"] = time.time()
        pass
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
    global log_file_path

    logger.log("\n")
    url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

    if any(path == flow.request.path for path in ["/api/chat", "/api/generate"]):
        log_file_path = generate_log_file_path(LOGS_DIR)

        logger.log("Log File Path:", log_file_path, colors=["GRAY", "INFO"])
    else:
        log_file_path = None

    logger.info(f"URL: {url}")
    # Log the serialized data as a JSON string
    request_dict = make_serializable(flow.request.data)
    logger.log(f"REQUEST KEYS:", list(
        request_dict.keys()), colors=["GRAY", "INFO"])
    logger.log(f"REQUEST:")
    logger.debug(json.dumps(request_dict.get('content', {}), indent=2))
    start_times[flow.id] = time.time()  # Store the start time for the request


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    global log_file_path
    global chunks

    logger.log("\n")
    # Log the serialized data as a JSON string
    response_dict = make_serializable(flow.response.data)
    logger.log(f"RESPONSE KEYS:", list(
        response_dict.keys()), colors=["GRAY", "INFO"])
    logger.log(f"RESPONSE:")
    logger.success("".join(chunks))

    end_time = time.time()  # Record the end time
    # if "stream" in start_times:
    #     end_time = time.time()
    #     time_taken = end_time - start_times["stream"]
    #     logger.log("\n\nStream took:", f"{time_taken:.2f} seconds", colors=[
    #         "LOG",
    #         "BRIGHT_SUCCESS",
    #     ])

    if flow.id in start_times:
        time_taken = end_time - start_times[flow.id]
        logger.log("Request total time took:", f"{time_taken:.2f} seconds", colors=[
            "LOG",
            "BRIGHT_SUCCESS",
        ])
        del start_times[flow.id]  # Clean up to avoid memory issues
    else:
        logger.warning(f"Start time for {flow.id} not found!")

    if log_file_path:
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
