import os
import shutil
from jet.file.utils import save_file
from jet.llm.mlx.remote import generation as gen
from typing import List, Dict, Callable
from jet.llm.mlx.remote.types import Message

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def llama_tool_example(query: str, tools: List[Callable]) -> None:
    """Demonstrate tool usage with Llama-3.2-3B-Instruct-4bit model."""
    print("=== Llama Chat Completion with Tools ===")

    messages: List[Message] = [
        {"role": "user", "content": query}
    ]
    response = gen.chat(
        messages=messages,
        model="mlx-community/llama-3.2-3b-instruct-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    save_file(response, f"{OUTPUT_DIR}/llama_tool_example.json")


def llama_no_tool_example(query: str) -> None:
    """Demonstrate no tools with Llama-3.2-3B-Instruct-4bit model."""
    print("=== Llama Chat Completion with Tools ===")

    messages: List[Message] = [
        {"role": "user", "content": query}
    ]
    response = gen.chat(
        messages=messages,
        model="mlx-community/llama-3.2-3b-instruct-4bit",
        max_tokens=100,
        verbose=True,
    )
    save_file(response, f"{OUTPUT_DIR}/llama_no_tool_example.json")


def mistral_tool_example(query: str, tools: List[Callable]) -> None:
    """Demonstrate tool usage with Mistral-7B-Instruct-v0.3-4bit model."""
    print("=== Mistral Chat Completion with Tools ===")

    messages: List[Message] = [
        {"role": "user", "content": query}
    ]
    response = gen.chat(
        messages=messages,
        model="mlx-community/mistral-7b-instruct-v0.3-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    save_file(response, f"{OUTPUT_DIR}/mistral_tool_example.json")


def main():
    query = "What is three thousand four hundred twenty three plus 6 thousand nine hundred ninety nine?"

    def add_two_numbers(a: int, b: int) -> int:
        """
        Add two numbers

        Args:
            a: The first number
            b: The second number

        Returns:
            int: The sum of the two numbers
        """
        return a + b

    tools = [add_two_numbers]

    print("=== Chat Completion Examples with Tools ===")
    llama_tool_example(query, tools)
    print("\n" + "="*50 + "\n")
    mistral_tool_example(query, tools)

    print("=== Chat Completion Examples no Tools ===")
    llama_no_tool_example(query)


if __name__ == "__main__":
    main()
