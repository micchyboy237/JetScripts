from typing import List, Type
from pydantic import BaseModel
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


class WeatherResponse(BaseModel):
    location: str
    temperature_celsius: float
    condition: str
    humidity_percent: int


def run_chat_structured_stream(
    messages: List[ChatMessage],
    response_model: Type[BaseModel],
    model: str = "qwen3-instruct-2507:4b",
    base_url: str = "http://shawn-pc.local:8080/v1",
    temperature: float = 0.0,
    verbose: bool = True,
) -> List[BaseModel]:
    """
    Stream **multiple** structured objects using a JSON **array**.
    Works with current llama.cpp (no json_lines support).
    """
    llm = LlamacppLLM(model=model, base_url=base_url, verbose=verbose)
    print("Streaming multiple weather reports...\n")

    # Wrap single model into a List[model] for array streaming
    from pydantic import TypeAdapter
    list_adapter = TypeAdapter(List[response_model])

    stream = llm.chat_structured_stream(
        messages=messages,
        response_model=list_adapter,          # <-- critical: List[WeatherResponse]
        temperature=temperature,
    )

    results: List[BaseModel] = []
    for item in stream:  # ← now yields only NEW WeatherResponse objects
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

    reports = run_chat_structured_stream(
        messages=structured_messages,
        response_model=WeatherResponse,
    )

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in reports:
        print(f"{r.location}: {r.temperature_celsius}°C, {r.condition}, {r.humidity_percent}% humidity")