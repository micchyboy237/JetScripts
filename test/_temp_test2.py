#!/usr/bin/env python3
"""
Send a streaming chat completion request to a local server and print each chunk
as it arrives, flushing stdout immediately.
"""

import json
import requests
from typing import Any, Callable
from jet.logger import logger

def _execute_tool(tool_call: dict[str, Any], available_functions: dict[str, Callable[..., Any]]) -> dict[str, Any]:
    """Run a single tool call and return a tool-response message."""
    func_name = tool_call["function"]["name"]
    args_str = tool_call["function"]["arguments"]
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.red(f"Invalid JSON arguments for {func_name}: {args_str}")
        return {
            "role": "tool",
            "content": json.dumps({"error": "Invalid arguments"}),
            "tool_call_id": tool_call.get("id", ""),
        }

    func = available_functions.get(func_name)
    if not func:
        logger.red(f"Tool {func_name} not implemented")
        return {
            "role": "tool",
            "content": json.dumps({"error": "Tool not found"}),
            "tool_call_id": tool_call.get("id", ""),
        }

    try:
        result = func(**args)
    except Exception as e:
        logger.red(f"Tool {func_name} raised: {e}")
        return {
            "role": "tool",
            "content": json.dumps({"error": str(e)}),
            "tool_call_id": tool_call.get("id", ""),
        }

    logger.green(f"Tool {func_name} → {result}")
    return {
        "role": "tool",
        "content": json.dumps({"result": result}),
        "tool_call_id": tool_call.get("id", ""),
    }

def main() -> None:
    url = "http://shawn-pc.local:8080/v1/chat/completions"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "acos",
                "description": "Return the arc cosine (measured in radians) of x.\n\nThe result is between 0 and pi.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acosh",
                "description": "Return the inverse hyperbolic cosine of x.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        },
    ]

    # ---- local implementations ------------------------------------------------
    import math
    available_functions: dict[str, Callable[..., Any]] = {
        "acos": lambda x: math.acos(float(x)),
        "acosh": lambda x: math.acosh(float(x)),
    }

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Use available tools to calculate arc cosine of 0.5."}],
        "model": "qwen3-instruct-2507:4b",
        "stream": True,
        "temperature": 0.0,
        "tools": tools,
    }
    headers = {"Content-Type": "application/json"}

    # ---------- first streaming request (assistant may emit tool_calls) ----------
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        tool_calls: list[dict] = []
        current_idx: int | None = None
        accumulated: dict[int, dict[str, str]] = {}

        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            data = line[6:]
            if data == b"[DONE]":
                break

            chunk = json.loads(data)
            choices = chunk.get("choices")
            if not choices:
                break

            delta = choices[0].get("delta", {})

            # ---- stream normal content ------------------------------------------------
            if content := delta.get("content"):
                logger.teal(content, flush=True)

            # ---- accumulate tool_call deltas -----------------------------------------
            if tc_deltas := delta.get("tool_calls"):
                for tc in tc_deltas:
                    idx = tc.get("index", 0)
                    if idx not in accumulated:
                        accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                        tool_calls.append({})
                        logger.magenta(f"[Tool {idx}] init", flush=True)

                    if tc_id := tc.get("id"):
                        accumulated[idx]["id"] += tc_id
                        logger.gray(f"[Tool {idx}] id: {tc_id!r}", flush=True)
                    if tc_fn := tc.get("function"):
                        if name := tc_fn.get("name"):
                            accumulated[idx]["name"] += name
                            logger.gray(f"[Tool {idx}] name: {name!r}", flush=True)
                        if args := tc_fn.get("arguments"):
                            accumulated[idx]["arguments"] += args
                            logger.gray(f"[Tool {idx}] args: {args!r}", flush=True)

        # build final tool_call objects
        for idx, acc in accumulated.items():
            tool_calls[idx] = {
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"]},
            }

    print()  # newline after streaming

    # ---------- if tool calls were emitted, execute them locally -----------------
    if tool_calls:
        messages = payload["messages"][:]
        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tc in tool_calls:
            messages.append(_execute_tool(tc, available_functions))

        # ---- final non-stream request to get the AI summary --------------------
        final_payload = {
            "model": payload["model"],
            "messages": messages,
            "temperature": payload["temperature"],
        }
        final_resp = requests.post(url, json=final_payload, headers=headers, timeout=60)
        final_resp.raise_for_status()
        final_json = final_resp.json()
        answer = final_json["choices"][0]["message"]["content"]
        logger.cyan("\nFinal AI response:")
        print(answer)
    else:
        logger.yellow("No tool calls emitted.")

if __name__ == "__main__":
    main()