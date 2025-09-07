import os
import shutil
from typing import List, Dict, TypedDict
from jet.file.utils import save_file
from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.remote.types import Message

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated",
                          os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def chat_text_response_example(query: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate chat completion with text response format."""
    print(f"=== {model.split('/')[-1]} Chat Completion with Text Response ===")
    messages: List[Message] = [{"role": "user", "content": query}]
    response = gen.chat(
        messages=messages,
        model=model,
        max_tokens=100,
        response_format="text",
        verbose=True,
    )
    print(f"Response: {response.get('content')}")
    save_file(response, f"{OUTPUT_DIR}/text_response_example.json")


def chat_json_response_example(query: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate chat completion with JSON response format."""
    print(f"=== {model.split('/')[-1]} Chat Completion with JSON Response ===")
    messages: List[Message] = [{"role": "user", "content": query}]
    response = gen.chat(
        messages=messages,
        model=model,
        max_tokens=100,
        response_format="json",
        verbose=True,
    )
    print(f"Response: {response.get('content')}")
    save_file(response, f"{OUTPUT_DIR}/json_response_example.json")


def chat_json_schema_response_example(query: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate chat completion with JSON schema response format."""
    print(f"=== {model.split('/')[-1]} Chat Completion with JSON Schema Response ===")

    class FactResult(TypedDict):
        fact: str
        source: str
    json_schema = {
        "type": "object",
        "properties": {
            "fact": {"type": "string"},
            "source": {"type": "string"}
        },
        "required": ["fact", "source"]
    }
    messages: List[Message] = [{"role": "user", "content": query}]
    response = gen.chat(
        messages=messages,
        model=model,
        max_tokens=100,
        response_format=json_schema,
        verbose=True,
    )
    print(f"Response: {response.get('content')}")
    save_file(response, f"{OUTPUT_DIR}/json_schema_response_example.json")


def main():
    query1 = "Tell me a fun fact about the universe."
    print("=== Example 1: Text Response ===")
    chat_text_response_example(query1)
    print("\n" + "="*50 + "\n")

    query2 = "Provide a fun fact about the universe in JSON format."
    print("=== Example 2: JSON Response ===")
    chat_json_response_example(query2)
    print("\n" + "="*50 + "\n")

    query3 = "Share a fun fact about the universe with its source in JSON format."
    print("=== Example 3: JSON Schema Response ===")
    chat_json_schema_response_example(query3)
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
