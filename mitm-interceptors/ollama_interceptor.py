import json
import os
import time
import traceback
from collections.abc import Iterable
from datetime import datetime
from jet.token.token_utils import token_counter
from jet.transformers.formatters import format_json
from mitmproxy import http
from jet.llm.llm_types import BaseGenerateResponse, OllamaChatResponse
from jet.transformers import make_serializable
from jet.logger import logger
from jet.file import save_file
from jet.utils.class_utils import get_class_name


LOGS_DIR = "jet-logs"
log_file_path = None

# Dictionary to store start times for requests
start_times: dict[str, float] = {}
chunks: list[str] = []


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
    messages = prompt_log_dict.get(
        'messages', prompt_log_dict.get('prompt', None))

    prompt_msgs = []
    if isinstance(messages, list):
        for item_idx, item in enumerate(messages):
            prompt_msg = (
                f"## Role\n{item.get('role')}\n\n"
                f"## Content\n{item.get('content')}\n"
            ).strip()
            prompt_msgs.append(
                f"### Message {item_idx + 1}\n\n```markdown\n{prompt_msg}\n```")
        prompt_log = "\n\n".join(prompt_msgs)
        prompt = messages[-1]['content']
    else:
        prompt_log = messages
        prompt = messages

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
    if final_dict.get("messages"):
        final_dict['messages'] = final_dict.pop('messages')
    if final_dict.get("prompt"):
        final_dict['prompt'] = final_dict.pop('prompt')
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
        f"## Prompt\n\n{prompt}\n\n"
        f"## JSON\n\n```json\n{json.dumps(final_dict, indent=2)}\n```\n\n"
    ).strip()
    return log_entry


def format_duration(nanoseconds: int) -> str:
    # Convert nanoseconds to seconds/minutes/milliseconds
    seconds = nanoseconds / 1e9
    formatted_time = f"{seconds:.2f}s"
    # if seconds >= 60:
    #     minutes = seconds / 60
    #     formatted_time = f"{minutes:.2f}m"
    # elif seconds >= 1:
    #     formatted_time = f"{seconds:.2f}s"
    # else:
    #     millis = seconds * 1000
    #     formatted_time = f"{millis:.2f}ms"
    return formatted_time


def get_response_durations(response_info: BaseGenerateResponse) -> dict[str, str]:
    durations_dict = {}
    durations = {
        k: v for k, v in response_info.items() if k.endswith('duration')}
    if durations:
        logger.info("Durations:")
        for key, value in durations.items():
            formatted_time = format_duration(value)
            durations_dict[key] = formatted_time
    return durations_dict


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

        # Check if "message" is a dictionary and if it contains the expected key
        if isinstance(chunk_dict.get("message"), dict):
            if chunk_dict["message"].get("role") == "assistant":
                content = chunk_dict["message"].get("content", "")
                if content:
                    chunks.append(content)
                    logger.success(content, flush=True)

        if chunk_dict.get("done"):
            chunks.append(chunk_dict)

    except json.JSONDecodeError:
        pass

    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    global log_file_path

    logger.newline()
    logger.log("request client_conn.id:", flow.client_conn.id,
               colors=["WHITE", "PURPLE"])

    # Store the start time for the request
    start_times[flow.id] = time.time()

    if any(path == flow.request.path for path in ["/api/embed", "/api/embeddings"]):
        request_dict = make_serializable(flow.request.data)
        logger.debug(f"REQUEST EMBEDDING:")
        logger.info(format_json(request_dict["content"]))

    elif any(path == flow.request.path for path in ["/api/chat"]):
        request_dict = make_serializable(flow.request.data)
        logger.log("\n")
        url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

        header_log_filename = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "log-filename"),
            None
        )
        header_event_start_time = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "event-start-time"),
            None
        )
        sub_dir = flow.request.path.replace("/", "-").strip("-")
        base_dir = os.path.join(header_log_filename, header_event_start_time)\
            if header_log_filename else flow.client_conn.id
        log_base_dir = os.path.join(sub_dir, base_dir)\
            if header_event_start_time else sub_dir
        log_file_path = generate_log_file_path(LOGS_DIR, log_base_dir)

        logger.info(f"URL: {url}")
        # Log the serialized data as a JSON string
        request_content: dict = request_dict["content"].copy()
        messages = request_content.pop("messages", None)
        options = request_content.pop("options", {})

        logger.newline()
        logger.gray("REQUEST MESSAGES:")
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

        token_count = token_counter(messages, request_content["model"])

        logger.newline()
        logger.log("STREAM:", request_content["stream"], colors=[
                   "GRAY", "INFO"])
        logger.log("MODEL:", request_content["model"], colors=["GRAY", "INFO"])
        logger.log("PROMPT LENGTH:", len(
            str(messages)), colors=["GRAY", "INFO"])
        logger.log("PROMPT TOKENS:", token_count, colors=["GRAY", "INFO"])

    elif any(path == flow.request.path for path in ["/api/generate"]):
        request_dict = make_serializable(flow.request.data)
        logger.log("\n")
        url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

        header_log_filename = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "log-filename"),
            None
        )
        header_event_start_time = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "event-start-time"),
            None
        )
        sub_dir = flow.request.path.replace("/", "-").strip("-")
        base_dir = os.path.join(header_log_filename, header_event_start_time)\
            if header_log_filename else flow.client_conn.id
        log_base_dir = os.path.join(sub_dir, base_dir)\
            if header_event_start_time else sub_dir
        log_file_path = generate_log_file_path(LOGS_DIR, log_base_dir)

        logger.info(f"URL: {url}")
        # Log the serialized data as a JSON string
        request_content: dict = request_dict["content"].copy()
        prompt = request_content.pop("prompt", None)

        logger.gray("REQUEST PROMPT:")
        logger.info(prompt)

        logger.log(f"REQUEST KEYS:", list(
            request_dict.keys()), colors=["GRAY", "INFO"])
        logger.log(f"REQUEST CONTENT KEYS:", list(
            request_dict["content"].keys()), colors=["GRAY", "INFO"])
        logger.log("REQUEST HEADERS:",
                   json.dumps(request_dict["headers"]), colors=["GRAY", "INFO"])

        logger.newline()
        logger.gray("REQUEST OPTIONS:")
        for key, value in options.items():
            logger.log(f"{key}:", value, colors=["GRAY", "DEBUG"])

        token_count = token_counter(prompt, request_content["model"])

        logger.newline()
        logger.log("PATH:", flow.request.path, colors=["GRAY", "INFO"])
        logger.log("STREAM:", request_content["stream"], colors=[
                   "GRAY", "INFO"])
        logger.log("MODEL:", request_content["model"], colors=["GRAY", "INFO"])
        logger.log("PROMPT LENGTH:", len(prompt), colors=["GRAY", "INFO"])
        logger.log("PROMPT TOKENS:", token_count, colors=["GRAY", "INFO"])

    else:
        log_file_path = None


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    global log_file_path
    global chunks

    logger.newline()
    logger.log("response client_conn.id:",
               flow.client_conn.id, colors=["WHITE", "PURPLE"])

    if any(path == flow.request.path for path in ["/api/chat", "/api/generate"]):
        logger.log("\n")
        # Get response info
        response_info = chunks.pop()
        if "context" in response_info:
            response_info.pop("context")
        # Log the serialized data as a JSON string
        response_dict: OllamaChatResponse = make_serializable(
            flow.response.data)

        if isinstance(response_dict, dict):
            logger.log(f"RESPONSE KEYS:", list(
                response_dict.keys()), colors=["GRAY", "INFO"])
        logger.log(f"RESPONSE INFO:", format_json(
            response_info), colors=["GRAY", "DEBUG"])

        durations = get_response_durations(response_info)
        total_duration = durations.pop("total_duration")
        for key, formatted_time in durations.items():
            logger.log(f"{key.title().replace("_", " ")}:", formatted_time,
                       colors=["WHITE", "DEBUG"])
        logger.log("Total Duration:", total_duration, colors=[
            "DEBUG",
            "SUCCESS",
        ])

        final_response_content = "".join(chunks)
        if response_info.get("response"):
            final_response_content = response_info.get("response")
        if not final_response_content:
            final_response_content = json.dumps(
                response_dict.get('content', {}), indent=1)

        # logger.log("RESPONSE:")
        # logger.debug(json.dumps(response_dict, indent=1))

        # logger.log("RESPONSE CONTENT:")
        # logger.success(final_response_content)

        logger.newline()
        logger.log("Response Text Length:", len(final_response_content),
                   colors=["DEBUG", "SUCCESS"])

        request_dict = make_serializable(flow.request.data)
        request_content: dict = request_dict["content"].copy()
        content_messages_key = "messages" if "/chat" in flow.request.path else "prompt"
        if not request_content["stream"]:
            logger.log("Response:",
                       final_response_content, colors=["DEBUG", "SUCCESS"])

        prompt_token_count = token_counter(
            request_content[content_messages_key], request_content["model"])
        response_token_count = token_counter(
            final_response_content, request_content["model"])
        total_tokens = prompt_token_count + response_token_count
        logger.newline()
        logger.log("Path:", flow.request.path, colors=["GRAY", "INFO"])
        logger.log("Stream:", request_content["stream"], colors=[
                   "GRAY", "INFO"])
        logger.log("Prompt Tokens:", prompt_token_count, colors=[
                   "WHITE", "DEBUG"])
        logger.log(
            "Response Tokens:",
            response_token_count,
            colors=["WHITE", "DEBUG"],
        )
        logger.log("Total Tokens:", total_tokens, colors=["DEBUG", "SUCCESS"])

        end_time = time.time()  # Record the end time

        if flow.id in start_times:
            time_taken = end_time - start_times[flow.id]
            logger.log("Request Time Took:", f"{time_taken:.2f} seconds", colors=[
                "SUCCESS",
                "BRIGHT_SUCCESS",
            ])
            del start_times[flow.id]  # Clean up to avoid memory issues
        else:
            logger.warning(f"Start time for {flow.id} not found!")

        logger.newline()

        if log_file_path:
            # Log prompt and response with metadata to the log file
            log_entry = generate_log_entry(flow)
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


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://localhost:11435 -p 11434
