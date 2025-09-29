import shutil
import json
import os
import threading
import time
import traceback
from collections.abc import Iterable
from jet.llm.models import OLLAMA_MODEL_CONTEXTS
from jet._token.token_utils import get_model_max_tokens, token_counter
from jet.transformers.formatters import format_json
from jet.transformers.json_parsers import parse_json
from mitmproxy import http
from jet.llm.llm_types import BaseGenerateResponse, OllamaChatResponse
from jet.transformers.object import make_serializable
from jet.logger import logger
from jet.file import save_file
from jet.utils.class_utils import get_class_name
from jet.utils.markdown import extract_block_content


LOGS_DIR = "ollama-logs"
log_file_path = None

# Dictionary to store start times for requests
start_times: dict[str, float] = {}
chunks: list[str] = []

# Global stop event
stop_event = threading.Event()


def remove_old_files_by_limit(base_dir: str, limit: int):
    """
    Removes the oldest files or directories in `base_dir` to maintain only `limit` most recent items.

    :param base_dir: The directory containing files and folders.
    :param limit: The maximum number of recent items to keep.
    """
    if not os.path.exists(base_dir):
        return

    existing_logs = sorted(
        (os.path.join(base_dir, f) for f in os.listdir(base_dir)),
        key=os.path.getctime
    )

    while len(existing_logs) > limit:
        oldest = existing_logs.pop(0)
        if os.path.isdir(oldest):
            shutil.rmtree(oldest)  # Remove directory and contents
        else:
            os.remove(oldest)  # Remove file


def generate_log_file_path(logs_dir, base_dir=None):
    # Determine the base directory
    if base_dir:
        log_dir = os.path.join(logs_dir, base_dir)
    else:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        # Create the log directory if it doesn't exist
        log_dir = os.path.join(base_dir, logs_dir)

    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamp and unique log file name
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_file_name = f"{timestamp}_{int(time.time())}.md"
    log_file_name = f"{int(time.time())}.md"
    log_file_path = os.path.realpath(os.path.join(
        log_dir, log_file_name).replace(' ', '_'))

    return log_file_path


def generate_log_entry(flow: http.HTTPFlow) -> str:
    """
    Generates a formatted log entry with metadata, prompt, and response.
    """
    global chunks
    local_chunks = chunks.copy()

    request_dict = make_serializable(flow.request.data)
    request_content: dict = request_dict["content"].copy()
    response_dict: OllamaChatResponse = make_serializable(flow.response.data)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    url = f"{flow.request.scheme}://{flow.request.host}{flow.request.path}"
    content_length = next(
        (field[1] for field in request_dict["headers"]["fields"] if field[0].lower() == "content-length"),
        None
    )
    log_filename = next(
        (field[1] for field in request_dict["headers"]["fields"] if field[0].lower() == "log-filename"),
        None
    )

    prompt_log = flow.request.data.content.decode('utf-8')
    prompt_log_dict = json.loads(prompt_log)
    model = prompt_log_dict['model']
    messages = prompt_log_dict.get('messages', prompt_log_dict.get('prompt', None))
    tools = request_content.get("tools")
    system = request_content.get("system")

    is_chat = isinstance(messages, list)
    has_tools = bool(tools)

    system = system or next(
        (field[1] for field in request_dict["headers"]["fields"] if field[0].lower() == "system"),
        None
    )

    if is_chat:
        messages_tuples = [(msg.get('role', ''), msg.get('content', '')) for msg in messages]
        updated_messages_tuples = update_system_roles(messages_tuples)
        messages = [{'role': role, 'content': content} for role, content in updated_messages_tuples]

    if is_chat:
        prompts = [message for message in messages if message.get('role', '').lower() != "system"]
        system_msg = (f"### System\n\n{system}\n").strip() if system else ""
        chat_msgs = [system_msg] if system else []
        for item in prompts:
            prompt_msg = (f"### {item.get('role', '')}\n\n{item.get('content')}").strip()
            chat_msgs.append(prompt_msg)
        prompt_log = "\n\n".join(chat_msgs)
        prompt = messages[-1]['content'] if messages else ""
    else:
        prompt_log = messages
        prompt = messages

    # Extract response content
    if request_content.get("stream", request_dict.get("stream", False)):
        final_response_content = "".join([chunk.get("content", "") for chunk in local_chunks])
        final_response_tool_calls = "".join(
            [json.dumps(chunk.get("tool_calls", ""), indent=1) for chunk in local_chunks])
        final_response_tool_calls = final_response_tool_calls.strip('"')
        if final_response_tool_calls:
            final_response_content += f"\n{final_response_tool_calls}".strip()
    else:
        final_response_content = response_dict.get("response", "") if "/generate" in flow.request.path else \
                                 response_dict.get("message", {}).get("content", "")
        if not final_response_content and isinstance(response_dict, dict):
            final_response_content = response_dict.get("content", "") or json.dumps(response_dict, indent=1)

    response = final_response_content
    final_dict_prompt = {
        **prompt_log_dict,
        'messages': messages if is_chat else None,
        'prompt': prompt if not is_chat else None,
    }
    final_dict_response = {"response": response}

    if final_dict_prompt.get("messages"):
        final_dict_prompt['messages'] = final_dict_prompt.pop('messages')
    if final_dict_prompt.get("prompt"):
        final_dict_prompt['prompt'] = final_dict_prompt.pop('prompt')
    if final_dict_prompt.get('tools'):
        final_dict_prompt['tools'] = final_dict_prompt.pop('tools')

    if response:
        response = extract_block_content(response)
        response = parse_json(response)
        response_type = "json" if isinstance(response, (dict, list)) else "markdown"
        response = f"```{response_type}\n{response if isinstance(response, str) else format_json(response)}\n```"

    response_str = f"## Response\n\n{response}\n\n" if response else ""
    if has_tools:
        formatted_tools = "".join(f"{idx}. {tool.get('function', {}).get('name', str(tool))}\n"
                                 for idx, tool in enumerate(tools, 1))
        tools_str = f"## Tools\n\n```text\n{formatted_tools.strip()}\n```\n\n"
    else:
        tools_str = ""
    prompt_log_str = f"## Prompts\n\n{prompt_log}\n\n" if is_chat else f"## Prompt\n\n{prompt}\n\n"

    prompt_token_count = next(
        (field[1] for field in request_dict["headers"]["fields"] if field[0].lower() == "tokens"),
        None
    ) or token_counter(messages, request_content.get("model", request_dict.get("model", None)))
    prompt_token_count = int(prompt_token_count)
    response_token_count = token_counter(final_response_content, request_content.get("model", request_dict.get("model", None)))
    tools_token_count = token_counter(tools, request_content.get("tools", request_dict.get("tools", None)))
    total_tokens = prompt_token_count + tools_token_count + response_token_count
    model_max_length = OLLAMA_MODEL_CONTEXTS[model]

    log_entry = (
        f"## Request Info\n\n"
        f"- **Log Filename**: {log_filename}\n"
        f"- **Is Chat:**: {'True' if is_chat else 'False'}\n"
        f"- **Has Tools:**: {'True' if has_tools else 'False'}\n"
        f"- **Stream**: {request_content.get('stream', request_dict.get('stream', False))}\n"
        f"- **Timestamp**: {timestamp}\n"
        f"- **Flow ID**: {flow.id}\n"
        f"- **URL**: {url}\n"
        f"- **Model**: {model}\n"
        f"- **Content length**: {content_length}\n"
        f"- **Prompt Tokens**: {prompt_token_count}\n"
        f"- **Response Tokens**: {response_token_count}\n"
        f"- **Total Tokens**: {total_tokens} / {model_max_length}\n"
        f"\n"
        f"{response_str}"
        f"{tools_str}"
        f"{prompt_log_str}"
        f"## JSON Request\n\n```json\n{format_json(final_dict_prompt)}\n```\n\n"
        f"## JSON Response\n\n```json\n{format_json(final_dict_response)}\n```\n\n"
    ).strip()
    return log_entry


def update_system_roles(messages):
    """
    Checks for multiple system roles in a list of Message (role, content) tuples.
    If more than one system role exists, updates all system roles except the first to assistant.

    Args:
        messages: List of tuples, each containing (role, content) where role is a string.

    Returns:
        Updated list of Message tuples with excess system roles changed to assistant.
    """
    # Count system roles
    system_count = sum(1 for role, _ in messages if role.lower() == "system")

    if system_count <= 1:
        return messages

    # Track if first system role has been encountered
    first_system_found = False
    updated_messages = []

    for role, content in messages:
        if role.lower() == "system":
            if not first_system_found:
                # Keep first system role
                updated_messages.append((role, content))
                first_system_found = True
            else:
                # Change subsequent system roles to assistant
                updated_messages.append(("assistant", content))
        else:
            # Keep non-system roles unchanged
            updated_messages.append((role, content))

    return updated_messages


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
    global stop_event

    if stop_event.is_set():
        logger.warning("Streaming stopped:", data)
        return b""

    if not data:  # Handle empty data
        return b""

    try:
        decoded_data = data.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning("Failed to decode data as UTF-8")
        return data

    try:
        chunk_dict = json.loads(decoded_data)
    except json.JSONDecodeError:
        # logger.warning("Failed to parse JSON from decoded data")
        return data

    # Ensure chunk_dict is a dictionary
    if not isinstance(chunk_dict, dict):
        logger.warning(
            f"Unexpected chunk type: {type(chunk_dict).__name__}")
        return data

    if not chunks:
        # logger.log("Stream started")
        # Store the start time for the stream
        # start_times["stream"] = time.time()
        pass

    try:
        # Check if "message" is a dictionary and if it contains the expected key
        if isinstance(chunk_dict.get("message"), dict):
            if chunk_dict["message"].get("role") == "assistant":
                content = chunk_dict["message"].get("content", "")
                chunks.append(chunk_dict["message"])
                logger.success(content, flush=True)
        elif isinstance(chunk_dict.get("response"), str):
            content = chunk_dict.get("response")
            chunks.append({"role": "assistant", "content": content})
            logger.success(content, flush=True)

        if chunk_dict.get("done"):
            chunks.append(chunk_dict)

    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        logger.debug(f"Chunk data: {chunk_dict}")

    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    global log_file_path
    global stop_event

    limit = 15

    logger.log("\n")
    url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"

    logger.newline()
    logger.log("request client_conn.id:", flow.client_conn.id,
               colors=["WHITE", "PURPLE"])

    logger.info(f"URL: {url}")

    # Store the start time for the request
    start_times[flow.id] = time.time()
    request_dict = make_serializable(flow.request.data)

    if stop_event.is_set():
        stop_event.clear()

    if "/stop" in flow.request.path:
        stop_event.set()
        flow.response = http.Response.make(400, b"Cancelled stream")
    elif any(path in flow.request.path for path in ["/embed", "/embeddings"]):
        request_content: dict = request_dict["content"].copy()
        input_texts = request_content.get("input", request_content.get("prompt", None))
        if isinstance(input_texts, list):
            token_count = sum(token_counter(item, request_content.get("model"))
                              for item in input_texts if isinstance(item, str))
        else:
            token_count = token_counter(
                input_texts, request_content.get("model"))

        logger.debug(f"EMBEDDING REQUEST: {flow.request.path}")
        logger.log("  Model: ", request_content.get("model"), colors=["GRAY", "TEAL"])
        logger.log("  Tokens: ", token_count, colors=["GRAY", "TEAL"])
        logger.log("  Max Tokens: ", get_model_max_tokens(request_content.get("model")), colors=["GRAY", "TEAL"])

        logger.newline()
        logger.log("REQUEST KEYS:")
        for k, v in request_dict.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
            else:
                logger.log(f"  {k}: <{type(v).__name__}>",
                           colors=["GRAY", "TEAL"])
        for key, value in request_dict["content"].items():
            logger.log(f"REQUEST CONTENT {key}:",
                       value, colors=["GRAY", "TEAL"])
        logger.log("REQUEST HEADERS:")
        headers = request_dict["headers"]
        if isinstance(headers, dict) and "fields" in headers:
            for k, v in headers["fields"]:
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
        else:
            logger.log(f"  {headers}", colors=["GRAY", "TEAL"])

    elif any(path in flow.request.path for path in ["/chat"]):
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
        if header_event_start_time:
            header_event_start_time = header_event_start_time.replace("|", "_")
            header_event_start_time = header_event_start_time.replace(":", "-")

        root_dir_path = flow.request.path.replace("/", "-").strip("-")
        if limit:
            remove_old_files_by_limit(
                os.path.join(LOGS_DIR, root_dir_path), limit)
        sub_dir_feature = header_log_filename or "_manual_call"
        sub_dir = os.path.join(root_dir_path, sub_dir_feature)
        base_dir = header_event_start_time if sub_dir_feature else flow.client_conn.id
        log_base_dir = os.path.join(sub_dir, base_dir)\
            if header_event_start_time else sub_dir

        if limit:
            remove_old_files_by_limit(os.path.join(LOGS_DIR, sub_dir), limit)
        log_file_path = generate_log_file_path(LOGS_DIR, log_base_dir)

        # Log the serialized data as a JSON string
        request_content: dict = request_dict["content"].copy()
        messages = request_content.pop("messages", None)
        tools = request_content.get("tools", None)
        options = request_content.pop("options", {})

        model_max_length = OLLAMA_MODEL_CONTEXTS[request_content['model']]

        logger.newline()
        logger.gray("REQUEST MESSAGES:")
        logger.info(format_json(messages) if not isinstance(
            messages, str) else messages)

        if tools:
            logger.newline()
            logger.gray("REQUEST TOOLS:")
            formatted_tools = ""
            for idx, tool in enumerate(tools, 1):
                tool_name = tool.get("function", {}).get(
                    "name") if isinstance(tool, dict) else str(tool)
                formatted_tools += f"{idx}. {tool_name}\n"
            logger.orange(formatted_tools.strip())

        logger.newline()
        logger.log("REQUEST KEYS:")
        for k, v in request_dict.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
            else:
                logger.log(f"  {k}: <{type(v).__name__}>",
                           colors=["GRAY", "TEAL"])
        for key, value in request_dict["content"].items():
            if key == "messages":
                logger.log(f"REQUEST CONTENT {key}: len={len(value) if value is not None else 0}", colors=[
                           "GRAY", "TEAL"])
            elif key == "tools":
                logger.log(f"REQUEST CONTENT {key}: len={len(value) if value is not None else 0}", colors=[
                           "GRAY", "TEAL"])
            else:
                logger.log(f"REQUEST CONTENT {key}:",
                           value, colors=["GRAY", "TEAL"])
        logger.log("REQUEST HEADERS:")
        headers = request_dict["headers"]
        if isinstance(headers, dict) and "fields" in headers:
            for k, v in headers["fields"]:
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
        else:
            logger.log(f"  {headers}", colors=["GRAY", "TEAL"])

        logger.newline()
        logger.gray("REQUEST OPTIONS:")
        for key, value in options.items():
            logger.log(f"{key}:", value, colors=["GRAY", "DEBUG"])

        token_count = token_counter(messages, request_content.get(
            "model", request_dict.get("model", None)))
        tools_token_count = token_counter(tools, request_content.get(
            "tools", request_dict.get("tools", None)))
        token_count += tools_token_count

        logger.newline()
        logger.log("STREAM:", request_content.get("stream", request_dict.get("stream", False)), colors=[
                   "GRAY", "INFO"])
        logger.log("MODEL:", request_content.get(
            "model", request_dict.get("model", None)), colors=["GRAY", "INFO"])
        logger.log("PROMPT TOKENS:", token_count, colors=["GRAY", "INFO"])
        logger.log("MAX TOKENS:", model_max_length, colors=["GRAY", "INFO"])
        logger.newline()

    elif any(path in flow.request.path for path in ["/generate"]):
        request_dict = make_serializable(flow.request.data)

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
        if header_event_start_time:
            header_event_start_time = header_event_start_time.replace("|", "_")

        root_dir_path = flow.request.path.replace("/", "-").strip("-")
        if limit:
            remove_old_files_by_limit(
                os.path.join(LOGS_DIR, root_dir_path), limit)
        sub_dir_feature = header_log_filename or "_manual_call"
        sub_dir = os.path.join(root_dir_path, sub_dir_feature)
        base_dir = header_event_start_time if sub_dir_feature else flow.client_conn.id
        log_base_dir = os.path.join(sub_dir, base_dir)\
            if header_event_start_time else sub_dir

        if limit:
            remove_old_files_by_limit(os.path.join(LOGS_DIR, sub_dir), limit)
        log_file_path = generate_log_file_path(LOGS_DIR, log_base_dir)

        # Log the serialized data as a JSON string
        request_content: dict = request_dict["content"].copy()
        prompt = request_content.pop("prompt", None)
        options = request_content.pop("options", {})

        model_max_length = OLLAMA_MODEL_CONTEXTS[request_content['model']]

        logger.gray("REQUEST PROMPT:")
        logger.info(prompt)

        logger.log("REQUEST KEYS:")
        for k, v in request_dict.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
            else:
                logger.log(f"  {k}: <{type(v).__name__}>",
                           colors=["GRAY", "TEAL"])
        for key, value in request_dict["content"].items():
            if key == "messages":
                logger.log(f"REQUEST CONTENT {key}: len={len(value) if value is not None else 0}", colors=[
                           "GRAY", "TEAL"])
            elif key == "tools":
                logger.log(f"REQUEST CONTENT {key}: len={len(value) if value is not None else 0}", colors=[
                           "GRAY", "TEAL"])
            else:
                logger.log(f"REQUEST CONTENT {key}:",
                           value, colors=["GRAY", "TEAL"])
        logger.log("REQUEST HEADERS:")
        headers = request_dict["headers"]
        if isinstance(headers, dict) and "fields" in headers:
            for k, v in headers["fields"]:
                logger.log(f"  {k}: {v}", colors=["GRAY", "TEAL"])
        else:
            logger.log(f"  {headers}", colors=["GRAY", "TEAL"])

        logger.gray("REQUEST OPTIONS:")
        for key, value in options.items():
            logger.log(f"{key}:", value, colors=["GRAY", "DEBUG"])

        token_count = token_counter(prompt, request_content.get(
            "model", request_dict.get("model", None)))

        logger.newline()
        logger.log("PATH:", flow.request.path, colors=["GRAY", "INFO"])
        logger.log("STREAM:", request_content.get("stream", request_dict.get("stream", False)), colors=[
                   "GRAY", "INFO"])
        logger.log("MODEL:", request_content.get(
            "model", request_dict.get("model", None)), colors=["GRAY", "INFO"])
        logger.log("PROMPT TOKENS:", token_count, colors=["GRAY", "INFO"])
        logger.log("MAX TOKENS:", model_max_length, colors=["GRAY", "INFO"])
        logger.newline()

    else:
        log_file_path = None


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    global log_file_path
    global chunks
    global stop_event
    local_chunks = chunks.copy()

    logger.newline()
    logger.log("response client_conn.id:",
               flow.client_conn.id, colors=["WHITE", "PURPLE"])

    # Check for error status codes in the response
    if flow.response and flow.response.status_code >= 400:
        error_type = "Client Error" if flow.response.status_code < 500 else "Server Error"
        try:
            reason = flow.response.data.reason.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            reason = str(flow.response)
        logger.error(
            f"Response Error: {error_type} - Status Code: {flow.response.status_code} - Reason: {reason}")
        logger.debug(f"Error Response Content: {flow.response.text}")
        chunks = []
        if flow.id in start_times:
            del start_times[flow.id]
        return

    if stop_event.is_set():
        logger.warning("Response - Cancelled stream")
        chunks = []
        return

    if any(path in flow.request.path for path in ["/chat", "/generate"]):
        logger.log("\n")
        request_dict = make_serializable(flow.request.data)
        request_content: dict = request_dict["content"].copy()
        response_dict: OllamaChatResponse = make_serializable(flow.response.data)

        # Extract response content
        final_response_content = ""
        if request_content.get("stream", request_dict.get("stream", False)):
            # Streaming case: Aggregate chunks
            final_response_content = "".join(
                [chunk.get("content", "") for chunk in local_chunks])
            final_response_tool_calls = "".join(
                [json.dumps(chunk.get("tool_calls", ""), indent=1) for chunk in local_chunks])
            final_response_tool_calls = final_response_tool_calls.strip('"')
            if final_response_tool_calls:
                final_response_content += f"\n{final_response_tool_calls}".strip()
        else:
            # Non-streaming case: Extract directly from response_dict
            if "/generate" in flow.request.path:
                final_response_content = response_dict.get("response", "")
            elif "/chat" in flow.request.path:
                final_response_content = response_dict.get("message", {}).get("content", "")

        # Fallback if no content is found
        if not final_response_content and isinstance(response_dict, dict):
            final_response_content = response_dict.get("content", "") or json.dumps(response_dict, indent=1)

        # Extract response_info for durations
        response_info = local_chunks[-1] if local_chunks else response_dict
        if isinstance(response_info, dict) and "context" in response_info:
            response_info = response_info.copy()
            response_info.pop("context", None)

        if isinstance(response_dict, dict):
            logger.log("RESPONSE KEYS:", list(
                response_dict.keys()), colors=["GRAY", "INFO"])
        logger.log("RESPONSE INFO:", format_json(
            response_info), colors=["GRAY", "DEBUG"])

        durations = get_response_durations(response_info)
        total_duration = durations.pop("total_duration", None)
        for key, formatted_time in durations.items():
            logger.log(f"{key.title().replace('_', ' ')}:", formatted_time,
                       colors=["WHITE", "DEBUG"])
        if total_duration is not None:
            logger.log("Total Duration:", total_duration, colors=[
                "DEBUG", "SUCCESS"])

        logger.newline()
        logger.log("Response Text Length:", len(final_response_content),
                   colors=["DEBUG", "SUCCESS"])

        if not request_content.get("stream", request_dict.get("stream", False)):
            logger.log("Response:", final_response_content, colors=["DEBUG", "SUCCESS"])

        if "/chat" in flow.request.path and response_dict.get("message", {}).get("tool_calls"):
            logger.log("Tools:", json.dumps(response_dict["message"].get("tool_calls", ""), indent=1),
                       colors=["DEBUG", "SUCCESS"])

        model_max_length = OLLAMA_MODEL_CONTEXTS[request_content['model']]
        messages = request_content.get('messages', request_content.get('prompt', None))
        tools = request_content.get("tools")
        prompt_token_count = next(
            (field[1] for field in request_dict["headers"]
             ["fields"] if field[0].lower() == "tokens"),
            None
        )
        if not prompt_token_count:
            prompt_token_count = token_counter(
                messages, request_content.get("model", request_dict.get("model", None)))
        prompt_token_count = int(prompt_token_count)
        response_token_count = token_counter(
            final_response_content, request_content.get("model", request_dict.get("model", None)))
        tools_token_count = token_counter(tools, request_content.get(
            "tools", request_dict.get("tools", None)))
        total_tokens = prompt_token_count + tools_token_count + response_token_count

        logger.newline()
        logger.log("Path:", flow.request.path, colors=["GRAY", "INFO"])
        logger.log("Stream:", request_content.get("stream", request_dict.get("stream", False)), colors=[
                   "GRAY", "INFO"])
        logger.log("Prompt Tokens:", prompt_token_count, colors=[
                   "WHITE", "DEBUG"])
        logger.log("Response Tokens:", response_token_count, colors=["WHITE", "DEBUG"])
        logger.log("Total Tokens:", f"{total_tokens} / {model_max_length}", colors=["DEBUG", "SUCCESS"])

        end_time = time.time()
        if flow.id in start_times:
            time_taken = end_time - start_times[flow.id]
            logger.log("Request Time Took:", f"{time_taken:.2f} seconds", colors=[
                "SUCCESS", "BRIGHT_SUCCESS"])
            del start_times[flow.id]
        else:
            logger.warning(f"Start time for {flow.id} not found!")

        logger.newline()

        if not stop_event.is_set() and log_file_path:
            log_entry = generate_log_entry(flow)
            save_file(log_entry, log_file_path)

    chunks = []


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


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://localhost:11435 -p 11434
