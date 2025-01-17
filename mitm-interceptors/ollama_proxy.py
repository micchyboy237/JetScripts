import json
import os
import time
from collections.abc import Iterable
from datetime import datetime
from jet.token.token_utils import token_counter
from jet.transformers.formatters import format_json
from mitmproxy import http
from jet.llm.llm_types import OllamaChatResponse
from jet.transformers import make_serializable
from jet.logger import logger
from jet.file import save_file


LOGS_DIR = "jet-logs"
log_file_path = None

# Dictionary to store start times for requests
start_times: dict[str, float] = {}
chunks: list[str] = []


def get_response_durations(response: OllamaChatResponse):
    durations = {
        k: v for k, v in response.items() if k.endswith('duration')}
    results = {
        "millis": 0,
        "seconds": 0,
        "minutes": 0,
    }
    if durations:
        logger.info("Durations:")
        for key, value in durations.items():
            # Convert nanoseconds to seconds/minutes/milliseconds
            seconds = value / 1e9
            if seconds >= 60:
                minutes = seconds / 60
                results["minutes"] = minutes
            elif seconds >= 1:
                results["seconds"] = seconds
            else:
                millis = seconds * 1000
                results["millis"] = millis
    return results


def generate_log_file_path(logs_dir, base_dir=None, limit=None):
    # Determine the base directory
    if base_dir:
        log_dir = os.path.join(logs_dir, base_dir)
    else:
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

    if limit:
        while len(existing_logs) >= limit:
            os.remove(existing_logs.pop(0))

    # Generate a timestamp and unique log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{timestamp}_{int(time.time())}.md"

    return os.path.realpath(os.path.join(log_dir, log_file_name))


def generate_log_entry(flow: http.HTTPFlow) -> str:
    """
    Generates a formatted log entry with metadata, prompt, and response.
    """
    global chunks

    request_dict = make_serializable(flow.request.data)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    url = f"{flow.request.scheme}://{flow.request.host}{flow.request.path}"
    # Handle potential StopIteration with a default value
    content_length = next(
        (field[1] for field in request_dict["headers"]
         ["fields"] if field[0].lower() == "content-length"),
        None
    )
    token_count = next(
        (field[1] for field in request_dict["headers"]
         ["fields"] if field[0].lower() == "tokens"),
        None
    )
    log_filename = next(
        (field[1] for field in request_dict["headers"]
         ["fields"] if field[0].lower() == "log-filename"),
        None
    )

    # Get last user prompt
    prompt_log = flow.request.data.content.decode('utf-8')
    prompt_log_dict = json.loads(prompt_log)
    model = prompt_log_dict['model']
    messages = prompt_log_dict['messages']

    prompt_msgs = []
    for item_idx, item in enumerate(messages):
        prompt_msg = (
            f"## Role\n{item.get('role')}\n\n"
            f"## Content\n{item.get('content')}\n"
        ).strip()
        prompt_msgs.append(
            f"### Message {item_idx + 1}\n\n```markdown\n{prompt_msg}\n```")

    prompt_log = "\n\n".join(prompt_msgs)

    # Get last assistant response
    contents = []
    for chunk in chunks:
        try:
            contents.append(chunk)
        except json.JSONDecodeError:
            pass
    response = "".join(contents)

    final_dict = {
        **prompt_log_dict,
        "response": response,
    }
    # Move 'messages' and 'response' to the end
    final_dict['messages'] = final_dict.pop('messages')
    final_dict['response'] = final_dict.pop('response')

    log_entry = (
        f"## Request Info\n\n"
        f"- **Timestamp**: {timestamp}\n"
        f"- **Flow ID**: {flow.id}\n"
        f"- **URL**: {url}\n"
        f"- **Model**: {model}\n"
        f"- **Content length**: {content_length}\n"
        f"- **Tokens**: {token_count}\n"
        f"- **Log Filename**: {log_filename}\n"
        f"\n"
        # f"## Messages ({len(messages)})\n\n{prompt_log}\n\n"
        f"## Response\n\n{response}\n\n"
        f"## Prompt\n\n{messages[-1]['content']}\n\n"
        f"## JSON\n\n```json\n{json.dumps(final_dict, indent=2)}\n```\n\n"
    ).strip()
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
            # logger.success(content, flush=True)
    except json.JSONDecodeError:
        pass

    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    global log_file_path

    if any(path == flow.request.path for path in ["/api/embed", "/api/embeddings"]):
        request_dict = make_serializable(flow.request.data)
        logger.log(f"REQUEST EMBEDDING:")
        logger.debug(json.dumps(format_json(request_dict), indent=1))

    if any(path == flow.request.path for path in ["/api/chat", "/api/generate"]):
        request_dict = make_serializable(flow.request.data)
        logger.log("\n")
        url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

        header_log_filename = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "log-filename"),
            None
        )
        log_file_path = generate_log_file_path(LOGS_DIR, header_log_filename)

        logger.info(f"URL: {url}")
        # Log the serialized data as a JSON string
        request_content: dict = request_dict["content"].copy()
        messages = request_content.pop(
            "messages", request_content.pop("prompt", None))
        options = request_content.pop("options", {})

        logger.newline()
        logger.gray("REQUEST PROMPT:")
        logger.info(format_json(messages) if not isinstance(
            messages, str) else messages)

        logger.log(f"REQUEST KEYS:", list(
            request_dict.keys()), colors=["GRAY", "INFO"])
        logger.log(f"REQUEST CONTENT KEYS:", list(
            request_dict["content"].keys()), colors=["GRAY", "INFO"])
        logger.log("REQUEST HEADERS:",
                   json.dumps(request_dict["headers"]), colors=["GRAY", "INFO"])

        logger.gray("REQUEST OPTIONS:")
        logger.debug(format_json(options))

        token_count = token_counter(
            request_dict["content"]["messages"], request_content["model"])

        logger.newline()
        logger.log("MODEL:", request_content["model"], colors=["GRAY", "INFO"])
        logger.log("PROMPT LENGTH:", len(
            str(request_dict["content"]["messages"])), colors=["GRAY", "INFO"])
        logger.log("PROMPT TOKENS:", token_count, colors=["GRAY", "INFO"])

        # Store the start time for the request
        start_times[flow.id] = time.time()
    else:
        log_file_path = None


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    global log_file_path
    global chunks

    if any(path == flow.request.path for path in ["/api/chat", "/api/generate"]):
        logger.log("\n")
        # Log the serialized data as a JSON string
        response_dict: OllamaChatResponse = make_serializable(
            flow.response.data)

        if isinstance(response_dict, dict):
            logger.log(f"RESPONSE KEYS:", list(
                response_dict.keys()), colors=["GRAY", "INFO"])

        final_response_content = "".join(chunks)
        if not final_response_content:
            final_response_content = json.dumps(
                response_dict.get('content', {}), indent=1)

        # logger.log("RESPONSE:")
        # logger.debug(json.dumps(response_dict, indent=1))

        # logger.log("RESPONSE CONTENT:")
        # logger.success(final_response_content)

        logger.log("\nRESPONSE LENGTH:", len(final_response_content),
                   colors=["WHITE", "BRIGHT_SUCCESS"])

        request_dict = make_serializable(flow.request.data)
        request_content: dict = request_dict["content"].copy()

        prompt_token_count = token_counter(
            request_dict["content"]["messages"], request_content["model"])
        response_token_count = token_counter(
            final_response_content, request_content["model"])
        logger.newline()
        logger.log("\nPROMPT TOKENS:", prompt_token_count, colors=[
                   "GRAY", "SUCCESS"])

        logger.log(
            "\nRESPONSE TOKENS:",
            response_token_count,
            colors=["GRAY", "SUCCESS"],
        )
        total_tokens = prompt_token_count + response_token_count
        logger.log("\nTOTAL:", total_tokens, colors=["WHITE", "SUCCESS"])

        durations = get_response_durations(response_dict)
        if durations.get("millis"):
            logger.log(f"Millis:", f"{durations["millis"]:.2f}ms", colors=[
                "WHITE", "LIME"])
        if durations.get("seconds"):
            logger.log(f"Seconds:", f"{durations["seconds"]:.2f}s", colors=[
                "WHITE", "WARNING"])
        if durations.get("minutes"):
            logger.log(f"Minutes:", f"{durations["minutes"]:.2f}m", colors=[
                "WHITE", "ORANGE"])

        end_time = time.time()  # Record the end time

        if flow.id in start_times:
            time_taken = end_time - start_times[flow.id]
            logger.log("Request total time took:", f"{time_taken:.2f} seconds", colors=[
                "LOG",
                "BRIGHT_SUCCESS",
            ])
            del start_times[flow.id]  # Clean up to avoid memory issues
        else:
            logger.warning(f"Start time for {flow.id} not found!")

        logger.log("Log File Path:", log_file_path, colors=["GRAY", "INFO"])

        if log_file_path:
            logger.log("FIRST 5 CHUNKS:")
            logger.debug(chunks[0:5])

            # Log prompt and response with metadata to the log file
            log_entry = generate_log_entry(flow)
            logger.info("Log Entry Result:")
            logger.success(log_entry)
            save_file(log_entry, log_file_path)

    chunks = []  # Clean up to avoid memory issues


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
    logger.warning(type(error))
    logger.error("Error occured on mitmproxy")
    logger.error(flow.error)


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://jetairm1:11435 -p 11434
