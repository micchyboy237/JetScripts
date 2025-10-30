#!/usr/bin/env python3
"""
Send a streaming chat completion request to a local server and print each chunk
as it arrives, flushing stdout immediately.
"""

import json
from typing import Any

from jet.logger import logger
import requests


def main() -> None:
    url = "http://shawn-pc.local:8080/v1/chat/completions"

    payload: dict[str, Any] = {
        "messages": [
            {
                "content": "Use available tools to calculate arc cosine of 0.5.",
                "role": "user",
            }
        ],
        "model": "qwen3-instruct-2507:4b",
        "stream": True,
        "temperature": 0.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "acos",
                    "description": "Return the arc cosine (measured in radians) of x.\n\nThe result is between 0 and pi.",
                    "parameters": {
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"],
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "acosh",
                    "description": "Return the inverse hyperbolic cosine of x.",
                    "parameters": {
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"],
                        "type": "object",
                    },
                },
            },
        ],
    }

    headers = {"Content-Type": "application/json"}

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            # Server sends lines prefixed with "data: "
            if line.startswith(b"data: "):
                data = line[6:]  # strip "data: "
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices")
                if not choices:
                    break
                # Extract the content delta (OpenAI-compatible format)
                delta = (
                    choices[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta:
                    logger.teal(delta, flush=True)
    print()  # final newline


if __name__ == "__main__":
    main()