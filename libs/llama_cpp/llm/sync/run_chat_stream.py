from typing import List
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage

def run_chat_stream(
    messages: List[ChatMessage],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    verbose: bool = True,
) -> None:
    """Synchronous streaming chat example with real-time token printing."""
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming response:")
    response_text = ""
    chunks = list(llm.chat_stream(messages, temperature=temperature, max_tokens=max_tokens))
    response_text = "".join(chunk.choices[0].delta.content for chunk in chunks)
    print("\n--- Stream complete ---\n")
    return response_text


if __name__ == "__main__":
    example_messages: List[ChatMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ]
    result = run_chat_stream(example_messages)
    print(f"Result type: {type(result)}")
