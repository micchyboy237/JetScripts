from typing import AsyncIterator, List
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage
from openai.types.chat import ChatCompletion


async def run_achat_stream(
    messages: List[ChatMessage],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    verbose: bool = True,
) -> str:
    """
    Asynchronous streaming chat example with real-time token printing.

    Returns:
        str: Full concatenated response text after stream completes.
    """
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming response:")
    response_text = ""
    stream: AsyncIterator[ChatCompletion] = llm.achat_stream(
        messages, temperature=temperature, max_tokens=max_tokens
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            content: str = chunk.choices[0].delta.content
            response_text += content
    print("\n--- Stream complete ---\n")
    return response_text


if __name__ == "__main__":
    import asyncio

    example_messages: List[ChatMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ]

    async def main() -> None:
        result = await run_achat_stream(example_messages)
        print(f"Result type: {type(result)}")
        print(f"Full response:\n{result}")

    asyncio.run(main())