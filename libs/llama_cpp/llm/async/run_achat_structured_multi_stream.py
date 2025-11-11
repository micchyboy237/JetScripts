import asyncio
from typing import List, Type
from pydantic import BaseModel, TypeAdapter
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


class WeatherResponse(BaseModel):
    location: str
    temperature_celsius: float
    condition: str
    humidity_percent: int


async def run_achat_structured_multi_stream(
    messages: List[ChatMessage],
    response_model: Type[BaseModel],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.0,
    verbose: bool = True,
) -> List[BaseModel]:
    """
    Async stream **multiple** structured objects using a JSON **array**.
    Works with current llama.cpp (no json_lines support).
    """
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming multiple weather reports...\n")
    list_adapter = TypeAdapter(List[response_model])
    stream = llm.achat_structured_stream(
        messages=messages,
        response_model=list_adapter,
        temperature=temperature,
    )
    results: List[BaseModel] = []
    async for item in stream:
        print(item)
        results.append(item)
    print(f"\nCompleted. Received {len(results)} weather report(s).")
    return results


if __name__ == "__main__":
    structured_messages: List[ChatMessage] = [
        {
            "role": "system",
            "content": """
You are a weather assistant for the Philippines.
Respond with a **single valid JSON array** containing one or more weather objects.
Each object must exactly match the schema.
Do NOT add explanations, markdown, or extra text.
Valid example:
[
  {"location": "Manila", "temperature_celsius": 29.5, "condition": "partly cloudy", "humidity_percent": 78},
  {"location": "Cebu", "temperature_celsius": 28.0, "condition": "sunny", "humidity_percent": 65},
  {"location": "Davao", "temperature_celsius": 30.2, "condition": "rainy", "humidity_percent": 82}
]
""".strip(),
        },
        {
            "role": "user",
            "content": "Current weather in major Philippine cities right now?",
        },
    ]
    reports = asyncio.run(
        run_achat_structured_multi_stream(
            messages=structured_messages,
            response_model=WeatherResponse,
        )
    )
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in reports:
        print(f"{r.location}: {r.temperature_celsius}°C, {r.condition}, {r.humidity_percent}% humidity")