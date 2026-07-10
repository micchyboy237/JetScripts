import json
import os

from headroom.client import HeadroomClient
from headroom.providers.openai import OpenAIProvider
from openai import OpenAI

original_client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)

client = HeadroomClient(
    original_client=original_client,
    provider=OpenAIProvider(),
    default_mode="optimize",
)

# Simulate a conversation with large tool outputs
messages = [
    {"role": "system", "content": "You analyze search results."},
    {"role": "user", "content": "Search for Python tutorials."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "python"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        # This is where Headroom shines - compressing large outputs
        "content": json.dumps(
            {
                "results": [
                    {"title": f"Result {i}", "score": 100 - i} for i in range(500)
                ]
            }
        ),
    },
    {"role": "user", "content": "What are the top 3 results?"},
]

# Headroom compresses the 500 results to ~20, keeping the most relevant
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

print(response.choices[0].message.content)
