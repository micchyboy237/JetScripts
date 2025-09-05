import json

from jet.logger import logger
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

# Specify the checkpoint
checkpoint = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# checkpoint = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

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
logger.gray("Prompt:")
logger.debug(tokenizer.decode(prompt))

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
