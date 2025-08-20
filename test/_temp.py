# Copyright © 2025 Apple Inc.

import json

from jet.logger import logger
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

# Specify the checkpoint
checkpoint = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)


SYSTEM_MESSAGE = "You are an AI assistant with MCP tools:\nTool: read_file\nDescription: Read the contents of a file.\nInput Schema: {\n  \"$defs\": {\n    \"FileInput\": {\n      \"properties\": {\n        \"file_path\": {\n          \"description\": \"Path to the file (e.g., 'example.txt')\",\n          \"title\": \"File Path\",\n          \"type\": \"string\"\n        },\n        \"encoding\": {\n          \"default\": \"utf-8\",\n          \"description\": \"File encoding\",\n          \"enum\": [\n            \"utf-8\",\n            \"ascii\"\n          ],\n          \"title\": \"Encoding\",\n          \"type\": \"string\"\n        }\n      },\n      \"required\": [\n        \"file_path\"\n      ],\n      \"title\": \"FileInput\",\n      \"type\": \"object\"\n    }\n  },\n  \"properties\": {\n    \"arguments\": {\n      \"$ref\": \"#/$defs/FileInput\"\n    }\n  },\n  \"required\": [\n    \"arguments\"\n  ],\n  \"title\": \"read_fileArguments\",\n  \"type\": \"object\"\n}\nOutput Schema: {\n  \"properties\": {\n    \"content\": {\n      \"description\": \"File contents or error message\",\n      \"title\": \"Content\",\n      \"type\": \"string\"\n    }\n  },\n  \"required\": [\n    \"content\"\n  ],\n  \"title\": \"FileOutput\",\n  \"type\": \"object\"\n}\n\nTool: navigate_to_url\nDescription: Navigate to a URL and return the page title, links from the same server, and all visible text content.\nInput Schema: {\n  \"$defs\": {\n    \"UrlInput\": {\n      \"properties\": {\n        \"url\": {\n          \"description\": \"URL to navigate to (e.g., 'https://example.com')\",\n          \"pattern\": \"^https?://\",\n          \"title\": \"Url\",\n          \"type\": \"string\"\n        }\n      },\n      \"required\": [\n        \"url\"\n      ],\n      \"title\": \"UrlInput\",\n      \"type\": \"object\"\n    }\n  },\n  \"properties\": {\n    \"arguments\": {\n      \"$ref\": \"#/$defs/UrlInput\"\n    }\n  },\n  \"required\": [\n    \"arguments\"\n  ],\n  \"title\": \"navigate_to_urlArguments\",\n  \"type\": \"object\"\n}\nOutput Schema: {\n  \"properties\": {\n    \"title\": {\n      \"description\": \"Page title or error message\",\n      \"title\": \"Title\",\n      \"type\": \"string\"\n    },\n    \"nav_links\": {\n      \"anyOf\": [\n        {\n          \"items\": {\n            \"type\": \"string\"\n          },\n          \"type\": \"array\"\n        },\n        {\n          \"type\": \"null\"\n        }\n      ],\n      \"default\": null,\n      \"description\": \"List of links from the same server\",\n      \"title\": \"Nav Links\"\n    },\n    \"text_content\": {\n      \"anyOf\": [\n        {\n          \"type\": \"string\"\n        },\n        {\n          \"type\": \"null\"\n        }\n      ],\n      \"default\": null,\n      \"description\": \"All visible text content on the page\",\n      \"title\": \"Text Content\"\n    }\n  },\n  \"required\": [\n    \"title\"\n  ],\n  \"title\": \"UrlOutput\",\n  \"type\": \"object\"\n}\n\nTool: summarize_text\nDescription: Summarize text content to a specified word limit.\nInput Schema: {\n  \"$defs\": {\n    \"SummarizeTextInput\": {\n      \"properties\": {\n        \"text\": {\n          \"description\": \"Text to summarize\",\n          \"title\": \"Text\",\n          \"type\": \"string\"\n        },\n        \"max_words\": {\n          \"default\": 100,\n          \"description\": \"Maximum number of words for the summary\",\n          \"maximum\": 500,\n          \"minimum\": 10,\n          \"title\": \"Max Words\",\n          \"type\": \"integer\"\n        }\n      },\n      \"required\": [\n        \"text\"\n      ],\n      \"title\": \"SummarizeTextInput\",\n      \"type\": \"object\"\n    }\n  },\n  \"properties\": {\n    \"arguments\": {\n      \"$ref\": \"#/$defs/SummarizeTextInput\"\n    }\n  },\n  \"required\": [\n    \"arguments\"\n  ],\n  \"title\": \"summarize_textArguments\",\n  \"type\": \"object\"\n}\nOutput Schema: {\n  \"properties\": {\n    \"summary\": {\n      \"description\": \"Summarized text\",\n      \"title\": \"Summary\",\n      \"type\": \"string\"\n    },\n    \"word_count\": {\n      \"description\": \"Number of words in the summary\",\n      \"title\": \"Word Count\",\n      \"type\": \"integer\"\n    }\n  },\n  \"required\": [\n    \"summary\",\n    \"word_count\"\n  ],\n  \"title\": \"SummarizeTextOutput\",\n  \"type\": \"object\"\n}\n\nTool: process_data\nDescription: Process data with progress.\nInput Schema: {\n  \"properties\": {\n    \"data\": {\n      \"title\": \"Data\",\n      \"type\": \"string\"\n    }\n  },\n  \"required\": [\n    \"data\"\n  ],\n  \"title\": \"process_dataArguments\",\n  \"type\": \"object\"\n}\nOutput Schema: {\n  \"properties\": {\n    \"result\": {\n      \"title\": \"Result\",\n      \"type\": \"string\"\n    }\n  },\n  \"required\": [\n    \"result\"\n  ],\n  \"title\": \"process_dataOutput\",\n  \"type\": \"object\"\n}\nUse JSON for tool requests: {'tool': 'name', 'arguments': {'arg': 'value'}}.\nFor chained tools, use placeholders like {{tool_name.output_field}} (e.g., {{navigate_to_url.text_content}})."

PROMPT = "Navigate to https://www.iana.org and summarize the text content in 100 words or less."


# An example tool, make sure to include a docstring and type hints
def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


tools = {"multiply": multiply}

# Specify the prompt and conversation history
prompt = "Multiply 12234585 and 48838483920."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=list(tools.values()),
    enable_thinking=False,
)

prompt_cache = make_prompt_cache(model)

# Generate the initial tool call:
# Expected response: A string containing a JSON tool call like
# "<tool_call>{'name': 'multiply', 'arguments': {'a': 12234585, 'b': 48838483920}}</tool_call>"
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

# Parse the tool call:
# (Note, the tool call format is model specific)
tool_open = "<tool_call>"
tool_close = "</tool_call>"
start_tool = response.find(tool_open) + len(tool_open)
end_tool = response.find(tool_close)
tool_call = json.loads(response[start_tool:end_tool].strip())
# Debug: Log the types and values of arguments
logger.debug(f"tool_call arguments: {tool_call['arguments']}")
logger.debug(
    f"Type of a: {type(tool_call['arguments']['a'])}, Value: {tool_call['arguments']['a']}")
logger.debug(
    f"Type of b: {type(tool_call['arguments']['b'])}, Value: {tool_call['arguments']['b']}")
# Expected tool_result: The product of 12234585 and 48838483920, which is 597573619473103572720
tool_result = tools[tool_call["name"]](**tool_call["arguments"])

logger.success(f"tool_result: {tool_result}")

# Put the tool result in the prompt with explicit instruction
messages = [
    {"role": "user", "content": "Multiply 12234585 and 48838483920."},
    {"role": "tool", "name": tool_call["name"], "content": str(tool_result)},
    {"role": "system", "content": "Confirm the result of the multiplication in a clear sentence."}
]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
)

# Create a new prompt cache to avoid context carryover
prompt_cache = make_prompt_cache(model)

# Generate the final response:
# Expected response: A string confirming the result, e.g., "The result of multiplying 12234585 and 48838483920 is 597573619473103572720."
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
