from typing import Optional
from jet.adapters.llama_cpp.llm import LlamacppLLM

def run_generate_stream(
    prompt: str,
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 500,
    verbose: bool = True,
) -> None:
    """Synchronous streaming generate example with real-time token printing."""
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming response:")
    chunks = list(llm.generate(prompt, temperature=temperature, max_tokens=max_tokens, stream=True))
    response_text = "".join(chunks)
    print("\n--- Stream complete ---\n")
    return response_text


if __name__ == "__main__":
    prompt = "Explain quantum computing in simple terms."
    result = run_generate_stream(prompt)
    print(f"Result type: {type(result)}")
