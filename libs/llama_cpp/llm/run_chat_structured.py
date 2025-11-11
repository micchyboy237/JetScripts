from typing import List
from pydantic import BaseModel
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


class WeatherResponse(BaseModel):
    location: str
    temperature_celsius: float
    condition: str
    humidity_percent: int


def run_chat_structured(
    messages: List[ChatMessage],
    response_model: type[BaseModel],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.0,
    verbose: bool = True,
) -> BaseModel:
    """Synchronous structured output using Pydantic model."""
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    result = llm.chat_structured(messages, response_model, temperature=temperature)
    print("Structured response:")
    print(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    structured_messages: List[ChatMessage] = [
        {
            "role": "system",
            "content": "You are a weather assistant. Always respond in valid JSON matching the schema.",
        },
        {"role": "user", "content": "What's the weather in Manila right now?"},
    ]
    run_chat_structured(structured_messages, WeatherResponse)