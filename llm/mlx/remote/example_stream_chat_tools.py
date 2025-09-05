from jet.llm.mlx.remote import generation as gen
from typing import List, Dict
from jet.llm.mlx.remote.types import Message


def llama_stream_tool_example() -> None:
    """Demonstrate streaming tool usage with Llama-3.2-3B-Instruct-4bit model."""
    print("=== Llama Streaming Chat Completion with Tools ===")
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    ]
    messages: List[Message] = [
        {"role": "user", "content": "What's the weather in London?", "tool_calls": []}
    ]
    for chunk in gen.stream_chat(
        messages=messages,
        model="mlx-community/llama-3.2-3b-instruct-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    ):
        if "choices" in chunk and chunk["choices"]:
            if chunk.get("tool_calls"):
                print(f"\nTool Calls: {chunk['tool_calls']}", flush=True)
    print("\n--- Llama Stream End ---")


def mistral_stream_tool_example() -> None:
    """Demonstrate streaming tool usage with Mistral-7B-Instruct-v0.3-4bit model."""
    print("=== Mistral Streaming Chat Completion with Tools ===")
    tools = [
        {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    ]
    messages: List[Message] = [
        {"role": "user", "content": "Calculate 2 + 2", "tool_calls": []}
    ]
    for chunk in gen.stream_chat(
        messages=messages,
        model="mlx-community/mistral-7b-instruct-v0.3-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    ):
        if "choices" in chunk and chunk["choices"]:
            if chunk.get("tool_calls"):
                print(f"\nTool Calls: {chunk['tool_calls']}")
    print("\n--- Mistral Stream End ---")


def main():
    print("=== Streaming Chat Completion Examples with Tools ===")
    llama_stream_tool_example()
    print("\n" + "="*50 + "\n")
    mistral_stream_tool_example()


if __name__ == "__main__":
    main()
