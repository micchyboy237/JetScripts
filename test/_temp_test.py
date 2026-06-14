import json
import os

from headroom import compress  # pip install "headroom-ai[all]"
from openai import OpenAI

# Local model config — matches your llama-server setup
LOCAL_MODEL = "Qwen3.5-0.8B-Q4_K_M"  # model string (llama-server ignores it, but required by client)
LOCAL_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1")

# Your llama-server has: -c 8192 context, -b 1024 batch, -ub 512 micro-batch
# Leave headroom for the system prompt, user message, and LLM output (~1500 tokens)
TOKEN_BUDGET = 5500  # safe headroom within 8192 ctx

# Example: Large tool output (e.g., search results or DB query)
# Reduced from 500 to 50 items — 500 items would already overflow 8192 tokens
large_tool_output = {
    "results": [
        {
            "id": i,
            "title": f"Item {i}",
            "description": f"Long description with details {i} "
            * 10,  # trimmed from *50
            "score": 100 - i,
        }
        for i in range(50)  # reduced from 500; compress() will handle the rest
    ],
    "metadata": {"total": 50, "query": "example search"},
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
        "content": json.dumps(large_tool_output),
    },
]

# === Compression Step ===
# Pass token_budget to hard-cap tokens for your 8192-ctx local model.
# model= here is used by headroom to pick compression strategy (not sent to llama-server).
result = compress(
    messages,
    model="gpt-4o",  # headroom uses this for strategy selection only
    token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
    ccr_enabled=True,  # reversible compression (default)
)

print(f"Tokens before: {result.tokens_before}")
print(f"Tokens after:  {result.tokens_after}")
print(f"Saved: {result.tokens_saved} tokens ({result.compression_ratio:.1%})")
print(f"Transforms: {result.transforms_applied}")

# Abort if still too large (safety check)
if result.tokens_after > TOKEN_BUDGET:
    raise RuntimeError(
        f"Compressed messages still too large: {result.tokens_after} tokens "
        f"(budget: {TOKEN_BUDGET}). Reduce input size or lower token_budget."
    )

# === Send to local llama-server ===
client = OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key="not-needed",  # llama-server doesn't require an API key
)

response = client.chat.completions.create(
    model=LOCAL_MODEL,
    messages=result.messages,  # Use the compressed version
    max_tokens=512,  # limit output tokens; your -ub 512 micro-batch aligns here
    temperature=0.7,
)

print("\nLLM Response:")
print(response.choices[0].message.content)
