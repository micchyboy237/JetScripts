import os
import shutil
from typing import TypedDict
from jet.file.utils import save_file
from jet.llm.mlx.remote import generation as gen

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated",
                          os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def stream_generate_text_response_example(prompt: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate streaming text generation with text response format."""
    print(
        f"=== Streaming {model.split('/')[-1]} Text Generation with Text Response ===")
    chunks = gen.stream_generate(
        prompt=prompt,
        model=model,
        max_tokens=100,
        response_format="text",
        verbose=True,
    )
    output = []
    for chunk in chunks:
        if chunk.get("content"):
            print(f"Content: {chunk['content']}")
        output.append(chunk)
    save_file(output, f"{OUTPUT_DIR}/text_response_stream_example.json")


def stream_generate_json_response_example(prompt: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate streaming text generation with JSON response format."""
    print(
        f"=== Streaming {model.split('/')[-1]} Text Generation with JSON Response ===")
    chunks = gen.stream_generate(
        prompt=prompt,
        model=model,
        max_tokens=100,
        response_format="json",
        verbose=True,
    )
    output = []
    for chunk in chunks:
        if chunk.get("content"):
            print(f"Content: {chunk['content']}")
        output.append(chunk)
    save_file(output, f"{OUTPUT_DIR}/json_response_stream_example.json")


def stream_generate_json_schema_response_example(prompt: str, model: str = "mlx-community/llama-3.2-3b-instruct-4bit") -> None:
    """Demonstrate streaming text generation with JSON schema response format."""
    print(
        f"=== Streaming {model.split('/')[-1]} Text Generation with JSON Schema Response ===")

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
    chunks = gen.stream_generate(
        prompt=prompt,
        model=model,
        max_tokens=100,
        response_format=json_schema,
        verbose=True,
    )
    output = []
    for chunk in chunks:
        if chunk.get("content"):
            print(f"Content: {chunk['content']}")
        output.append(chunk)
    save_file(output, f"{OUTPUT_DIR}/json_schema_response_stream_example.json")


def main():
    prompt1 = "Tell me a fun fact about the universe."
    print("=== Example 1: Text Response ===")
    stream_generate_text_response_example(prompt1)
    print("\n" + "="*50 + "\n")

    prompt2 = "Provide a fun fact about the universe in JSON format."
    print("=== Example 2: JSON Response ===")
    stream_generate_json_response_example(prompt2)
    print("\n" + "="*50 + "\n")

    prompt3 = "Share a fun fact about the universe with its source in JSON format."
    print("=== Example 3: JSON Schema Response ===")
    stream_generate_json_schema_response_example(prompt3)
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
