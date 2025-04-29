# Copyright © 2025 Apple Inc.

import json

from mlx_lm import generate, stream_generate, load
from mlx_lm.models.cache import make_prompt_cache
from jet.transformers.formatters import format_json
from jet.logger import logger

# Specify the checkpoint
checkpoint = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)


# An example tool, make sure to include a docstring and type hints
def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    logger.info(f"Called multiply. Params: (a={a}, b={b})")
    return a * b


tools = {"multiply": multiply}
logger.gray("Tools:")
logger.debug(tools)

# Specify the prompt and conversation history
prompt = "Multiply 12234585 and 48838483920."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tools=list(tools.values())
)

prompt_cache = make_prompt_cache(model)

# Generate the initial tool call:
# response = generate(
#     model=model,
#     tokenizer=tokenizer,
#     prompt=prompt,
#     max_tokens=2048,
#     verbose=True,
#     prompt_cache=prompt_cache,
# )
kwargs = {
    "max_tokens": 2048,
    "prompt_cache": prompt_cache,
}
response = ""
for chunk in stream_generate(model, tokenizer, prompt, **kwargs):
    logger.success(chunk.text, flush=True)
    response += chunk.text


tool_call = json.loads(response)
tool_result = tools[tool_call["name"]](**tool_call["parameters"])

logger.gray("Tool Result:")
logger.success(tool_result)

# Put the tool result in the prompt
messages = [{"role": "tool", "name": tool_call["name"], "content": tool_result}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Generate the final response:
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)
