import json
import re
from jet.llm.mlx.mlx_utils import create_tool_function
from jet.logger import logger
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache
from typing import Dict, Any, List, Optional, Callable
from functools import partial

checkpoint = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
model, tokenizer = load(path_or_hf_repo=checkpoint)

# Tool info provided by the user (unchanged)
tool_info = {
    "name": "navigate_to_url",
    "description": "Navigate to a URL and return the page title, links from the same server, and all visible text content.",
    "schema": {
        "$defs": {
            "UrlInput": {
                "properties": {
                    "url": {
                        "description": "URL to navigate to (e.g., 'https://example.com')",
                        "pattern": "^https?://",
                        "title": "Url",
                        "type": "string"
                    }
                },
                "required": [
                    "url"
                ],
                "title": "UrlInput",
                "type": "object"
            }
        },
        "properties": {
            "arguments": {
                "$ref": "#/$defs/UrlInput"
            }
        },
        "required": [
            "arguments"
        ],
        "title": "navigate_to_urlArguments",
        "type": "object"
    },
    "outputSchema": {
        "properties": {
            "url": {
                "description": "The URL that was navigated to",
                "title": "Url",
                "type": "string"
            },
            "title": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ],
                "default": None,
                "description": "Page title or error message",
                "title": "Title"
            },
            "nav_links": {
                "anyOf": [
                    {
                        "items": {
                            "type": "string"
                        },
                        "type": "array"
                    },
                    {
                        "type": "null"
                    }
                ],
                "default": None,
                "description": "List of links from the same server",
                "title": "Nav Links"
            },
            "text": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ],
                "default": None,
                "description": "All visible text content on the page",
                "title": "Text"
            }
        },
        "required": [
            "url"
        ],
        "title": "UrlOutput",
        "type": "object"
    }
}


# Rest of the code remains unchanged
tools = {tool_info["name"]: create_tool_function(tool_info)}

prompt = "Navigate to https://example.com and summarize the page content."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=[tools[tool_info["name"]]],
    enable_thinking=False,
)

prompt_cache = make_prompt_cache(model)
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

logger.gray("Response:")
logger.success(response)

tool_open = "<tool_call>"
tool_close = "</tool_call>"
start_tool = response.find(tool_open) + len(tool_open)
end_tool = response.find(tool_close)
tool_call = json.loads(response[start_tool:end_tool].strip())
logger.debug(f"tool_call arguments: {tool_call['arguments']}")
logger.debug(
    f"Type of url: {type(tool_call['arguments']['url'])}, Value: {tool_call['arguments']['url']}")

tool_result = tools[tool_call["name"]](**tool_call["arguments"])
logger.success(f"tool_result: {tool_result}")

messages = [
    {"role": "user", "content": "Navigate to https://example.com and summarize the page content."},
    {"role": "tool", "name": tool_call["name"],
        "content": json.dumps(tool_result)},
    {"role": "system", "content": "Summarize the content of the page based on the tool result in a clear sentence."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
)

prompt_cache = make_prompt_cache(model)
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

logger.gray("Response 2:")
logger.success(response)
