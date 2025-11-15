# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/libs/llama_cpp/llm/sync/run_chat_with_tools.py
from typing import List, Dict, Any, Callable
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def get_weather(city: str) -> str:
    """Get mock weather for a city."""
    weather_map = {
        "manila": "28°C, partly cloudy",
        "tokyo": "15°C, rainy",
        "new york": "8°C, clear"
    }
    return weather_map.get(city.lower(), "Unknown city")


def run_chat_with_tools(
    messages: List[ChatMessage],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    verbose: bool = True,
) -> str:
    """Synchronous non-streaming chat with tool calling."""
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    available_functions: Dict[str, Callable[..., Any]] = {
        "add": add,
        "get_weather": get_weather
    }

    print("Generating response with tools...")
    response = llm.chat_with_tools(
        messages=messages,
        tools=tools,
        available_functions=available_functions,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False  # Explicit for clarity
    )
    print("\n--- Final Response ---\n")
    print(response)
    return response


if __name__ == "__main__":
    example_messages: List[ChatMessage] = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What is 18 + 24? Also, what's the weather in Manila?"},
    ]

    result = run_chat_with_tools(example_messages)
    print(f"\nResult type: {type(result)}")