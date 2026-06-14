import json

from headroom import compress  # pip install "headroom-ai[all]"
from openai import OpenAI

# Example: Large tool output (e.g., search results or DB query)
large_tool_output = {
    "results": [
        {
            "id": i,
            "title": f"Item {i}",
            "description": f"Long description with details {i} " * 50,
            "score": 100 - i,
        }
        for i in range(500)  # Simulate 500 items
    ],
    "metadata": {"total": 500, "query": "example search"},
}

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Analyze tool outputs carefully.",
    },
    {"role": "user", "content": "Summarize the top results from this search."},
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps(large_tool_output),  # This is what gets huge
    },
]

# === Compression Step ===
result = compress(
    messages,
    model="gpt-4o",  # Helps choose optimal strategy
    # Optional: token_budget=8000, ccr_enabled=True (default)
)

print(f"Tokens before: {result.tokens_before}")
print(f"Tokens after:  {result.tokens_after}")
print(f"Saved: {result.tokens_saved} tokens ({result.compression_ratio:.1%})")
print(f"Transforms: {result.transforms_applied}")

# === Send to OpenAI ===
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=result.messages,  # Use the compressed version
)

print("\nLLM Response:")
print(response.choices[0].message.content)
