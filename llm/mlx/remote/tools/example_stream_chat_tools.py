import os
import shutil
from typing import List, Dict, Callable, TypedDict
from jet.file.utils import save_file
from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.remote.types import Message

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def stream_llama_tool_example(query: str, tools: List[Callable], model: str = "mlx-community/llama-3.2-3b-instruct-4bit"):
    """Demonstrate streaming tool usage with the specified model."""
    print(
        f"=== Streaming {model.split('/')[-1]} Chat Completion with Tools ===")
    messages: List[Message] = [{"role": "user", "content": query}]
    chunks = gen.stream_chat(
        messages=messages,
        model=model,
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    output = []
    for chunk in chunks:
        if chunk.get("content"):
            print(f"Content: {chunk['content']}")
        if chunk.get("tool_calls"):
            print(f"Tool Calls: {chunk['tool_calls']}")
        if chunk.get("tool_execution"):
            print(f"Tool Execution: {chunk['tool_execution']}")
        output.append(chunk)
    return output


def stream_mistral_tool_example(query: str, tools: List[Callable], model: str = "mlx-community/mistral-7b-instruct-v0.3-4bit"):
    """Demonstrate streaming tool usage with the specified model."""
    print(
        f"=== Streaming {model.split('/')[-1]} Chat Completion with Tools ===")
    messages: List[Message] = [{"role": "user", "content": query}]
    chunks = gen.stream_chat(
        messages=messages,
        model=model,
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    output = []
    for chunk in chunks:
        if chunk.get("content"):
            print(f"Content: {chunk['content']}")
        if chunk.get("tool_calls"):
            print(f"Tool Calls: {chunk['tool_calls']}")
        if chunk.get("tool_execution"):
            print(f"Tool Execution: {chunk['tool_execution']}")
        output.append(chunk)
    return output


def main():
    # Example 1: Simple addition tool
    def add_two_numbers(a: int, b: int) -> int:
        """
        Add two numbers.
        Args:
            a: The first number
            b: The second number
        Returns:
            int: The sum of the two numbers
        """
        return a + b

    query1 = "What is three thousand four hundred twenty three plus 6 thousand nine hundred ninety nine?"
    print("=== Example 1: Llama - Addition Tool ===")
    output = stream_llama_tool_example(query1, [add_two_numbers])
    save_file(
        output, f"{OUTPUT_DIR}/add_two_numbers_stream_tool_llama_example.json")
    print("\n" + "="*50 + "\n")

    # Example 2: Weather query tool
    class WeatherResult(TypedDict):
        city: str
        temperature: int
        unit: str
        condition: str

    def get_weather(city: str, unit: str) -> WeatherResult:
        """
        Get the weather for a specified city.

        Args:
            city: The city, e.g. San Francisco
            unit: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])

        Returns:
            WeatherResult: A dictionary with the following fields:
                temperature: int, the temperature in the specified unit
                condition: str, the weather condition (e.g., 'Sunny')
        """
        return {"city": city, "temperature": 20, "unit": unit, "condition": "Sunny"}

    query2 = "What's the weather like in Paris in Fahrenheit?"
    print("=== Example 2: Llama - Weather Tool ===")
    output = stream_llama_tool_example(query2, [get_weather])
    save_file(
        output, f"{OUTPUT_DIR}/get_weather_stream_tool_llama_example.json")
    print("\n" + "="*50 + "\n")

    # Example 3: No tool needed
    query3 = "Tell me a fun fact."
    print("=== Example 3: Llama - No Tool ===")
    output = stream_llama_tool_example(query3, [])
    save_file(
        output, f"{OUTPUT_DIR}/no_tool_stream_llama_example.json")
    print("\n" + "="*50 + "\n")

    print("=== Example 1: Mistral - Addition Tool ===")
    output = stream_mistral_tool_example(query1, [add_two_numbers])
    save_file(
        output, f"{OUTPUT_DIR}/add_two_numbers_stream_tool_mistral_example.json")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
