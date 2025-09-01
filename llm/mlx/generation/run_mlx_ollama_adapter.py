import os
import shutil
import argparse
from typing import Dict, List, Optional, Union, Callable
from ollama._types import Message, Options
from jet.llm.mlx.base import MLX
from jet.models.model_types import LLMModelType
from jet.llm.mlx.client import Message as MLXMessage, RoleMapping
from jet.file.utils import save_file
from jet.llm.mlx.mlx_utils import parse_tool_call
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Define logger
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)


def convert_ollama_to_mlx_messages(ollama_messages: Union[str, List[Message]]) -> Union[str, List[MLXMessage]]:
    """
    Convert Ollama messages to MLX-compatible message format.
    Args:
        ollama_messages: Either a string prompt or a list of Ollama Message objects.
    Returns:
        Either a string or a list of MLX Message objects compatible with MLX client.
    """
    if isinstance(ollama_messages, str):
        return ollama_messages
    elif isinstance(ollama_messages, list):
        mlx_messages = []
        for msg in ollama_messages:
            tool_calls = []
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_calls.append({
                        "type": call.get("type", "function"),
                        "function": {
                            "name": call.get("function", {}).get("name", ""),
                            "arguments": call.get("function", {}).get("arguments", {})
                        }
                    })
            mlx_msg = MLXMessage(
                role=msg.get("role", ""),
                content=msg.get("content", "") or "",
                images=msg.get("images", []),
                tool_name=msg.get("tool_name", None),
                tool_calls=tool_calls
            )
            mlx_messages.append(mlx_msg)
        return mlx_messages
    else:
        raise TypeError(
            "ollama_messages must be a string or a list of Message objects")


def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    Args:
        expression: A string representing a mathematical expression (e.g., "2 + 2").
    Returns:
        The result of the evaluated expression.
    """
    try:
        # Safe eval with restricted globals
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {str(e)}")
        raise


def run_mlx_ollama_adapter(
    messages: Union[str, List[Message]],
    model: Optional[LLMModelType] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    options: Optional[Options] = None,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    verbose: bool = False
):
    """
    Run MLX model with Ollama-compatible message input.
    Args:
        messages: Either a string prompt or a list of Ollama Message objects.
        model: Model to use for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        options: Ollama Options object for additional parameters.
        system_prompt: Optional system prompt to prepend.
        stream: Whether to stream the response.
        verbose: Enable verbose logging.
    Returns:
        MLX completion response or iterator of responses if streaming.
    """
    mlx = MLX(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
        log_dir=f"{OUTPUT_DIR}/chats"
    )
    if options:
        mlx.client.cli_args.max_tokens = options.num_predict or mlx.client.cli_args.max_tokens
        mlx.client.cli_args.temperature = options.temperature or mlx.client.cli_args.temperature
        mlx.client.cli_args.top_p = options.top_p or mlx.client.cli_args.top_p
        mlx.client.cli_args.top_k = options.top_k or mlx.client.cli_args.top_k
        mlx.client.cli_args.stop = options.stop or mlx.client.cli_args.stop
        mlx.client.cli_args.repetition_penalty = options.repeat_penalty or mlx.client.cli_args.repetition_penalty

    # Define available tools
    tools = [calculate]

    mlx_messages = convert_ollama_to_mlx_messages(messages)

    if stream:
        response = mlx.stream_chat(
            messages=mlx_messages,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            tools=tools
        )
        return response
    else:
        response = mlx.chat(
            messages=mlx_messages,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            tools=tools
        )

        # Check if response contains a tool call
        content = response.get("content", "")
        if "<tool_call>" in content:
            logger.gray("Processing tool call")
            tool_call = parse_tool_call(content)
            logger.debug(f"Tool call arguments: {tool_call['arguments']}")

            # Execute the tool
            tool_result = None
            if tool_call["name"] == "calculate":
                tool_result = calculate(**tool_call["arguments"])
                logger.success(f"Tool result: {tool_result}")

            # Append tool result to messages and generate confirmation
            if tool_result is not None:
                new_messages = messages if isinstance(messages, list) else [
                    {"role": "user", "content": messages}]
                new_messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": str(tool_result)
                })
                new_messages.append({
                    "role": "system",
                    "content": "Confirm the result of the calculation in a clear sentence."
                })

                # Convert updated messages to MLX format
                mlx_new_messages = convert_ollama_to_mlx_messages(new_messages)

                # Generate confirmation response
                confirmation_response = mlx.chat(
                    messages=mlx_new_messages,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    verbose=verbose
                )
                return confirmation_response

        return response


def ollama_mlx_model_mapping(ollama_model: str) -> LLMModelType:
    """
    Map Ollama model names to MLX-compatible model names.
    Args:
        ollama_model: The Ollama model name (e.g., 'llama3.2').
    Returns:
        The corresponding MLX model name (e.g., 'mlx-community/Llama-3.2-3B-Instruct-4bit').
    """
    model_map: Dict[str, LLMModelType] = {
        "llama3.2": "mlx-community/Llama-3.2-3B-Instruct-4bit"
    }
    return model_map.get(ollama_model, "mlx-community/Llama-3.2-3B-Instruct-4bit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLX with Ollama-compatible messages")
    parser.add_argument("--model", type=str, default="llama3.2",
                        help="Model to use (default: llama3.2)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum number of tokens (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--stream", action="store_true",
                        default=True, help="Stream the response (default: True)")
    parser.add_argument("--verbose", action="store_true",
                        default=False, help="Enable verbose logging (default: False)")
    args = parser.parse_args()
    mapped_model = ollama_mlx_model_mapping(args.model)

    logger.info(f"Logs: {log_file}")

    simple_prompt = "Hello, how can I assist you today?"
    print("\nExample 1: Simple string prompt")
    response = run_mlx_ollama_adapter(
        messages=simple_prompt,
        model=mapped_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        verbose=args.verbose
    )
    if args.stream:
        chunks = []
        for chunk in response:
            print(chunk["choices"][0]["message"]
                  ["content"], end="", flush=True)
            chunks.append(chunk)
        save_file(chunks, f"{OUTPUT_DIR}/simple_string_prompt_response.json")
    else:
        print(response["choices"][0]["message"]["content"])
        save_file(response, f"{OUTPUT_DIR}/simple_string_prompt_response.json")

    messages = [
        Message(role="system", content="You are a helpful AI assistant"),
        Message(role="user", content="What's the capital of France?")
    ]
    print("\nExample 2: System and user messages")
    response = run_mlx_ollama_adapter(
        messages=messages,
        model=mapped_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        verbose=args.verbose
    )
    if args.stream:
        chunks = []
        for chunk in response:
            print(chunk["choices"][0]["message"]
                  ["content"], end="", flush=True)
            chunks.append(chunk)
        save_file(chunks, f"{OUTPUT_DIR}/system_user_messages_response.json")
    else:
        print(response["choices"][0]["message"]["content"])
        save_file(response, f"{OUTPUT_DIR}/system_user_messages_response.json")

    messages_with_tools = [
        Message(
            role="user",
            content="Calculate 2 + 2",
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name="calculate",
                        arguments={"expression": "2 + 2"}
                    )
                )
            ]
        )
    ]
    print("\nExample 3: Messages with tool calls")
    response = run_mlx_ollama_adapter(
        messages=messages_with_tools,
        model=mapped_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        verbose=args.verbose
    )
    if args.stream:
        chunks = []
        for chunk in response:
            content = chunk["choices"][0]["message"]["content"]
            print(content, end="", flush=True)
            chunks.append(chunk)
        save_file(chunks, f"{OUTPUT_DIR}/tool_calls_response.json")
    else:
        content = response["choices"][0]["message"]["content"]
        print(content)
        save_file(response, f"{OUTPUT_DIR}/tool_calls_response.json")
