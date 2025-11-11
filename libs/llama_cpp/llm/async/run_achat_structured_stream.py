import asyncio
from typing import List
from pydantic import BaseModel
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


class WeatherResponse(BaseModel):
    location: str
    temperature_celsius: float
    condition: str
    humidity_percent: int


async def run_achat_structured_stream(
    messages: List[ChatMessage],
    response_model: type[BaseModel],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.0,
    verbose: bool = True,
) -> BaseModel:
    """Async stream structured outputs using Pydantic model and return first valid instance."""
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming structured response(s):")
    async for result in llm.achat_structured_stream(messages, response_model, temperature=temperature):
        print(result)
        return result
    


if __name__ == "__main__":
    structured_messages: List[ChatMessage] = [
        {
            "role": "system",
            "content": "You are a weather assistant. Always respond in valid JSON matching the schema.",
        },
        {"role": "user", "content": "What's the weather in Manila right now?"},
    ]
    result = asyncio.run(run_achat_structured_stream(structured_messages, WeatherResponse))
    print(f"Result type: {type(result)}")