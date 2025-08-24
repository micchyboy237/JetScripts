import os
import shutil
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.llm.mlx.mlx_utils import parse_tool_call
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# Specify the checkpoint
checkpoint = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

# Load the corresponding model and tokenizer
model = MLXModelRegistry.load_model(checkpoint, log_dir=f"{OUTPUT_DIR}/chats")

# An example tool, make sure to include a docstring and type hints


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


base_tools = {"multiply": multiply}

# Specify the prompt and conversation history
prompt = "Multiply 12234585 and 48838483920."
messages = [{"role": "user", "content": prompt}]

tools = list(base_tools.values())
chat_template_args: ChatTemplateArgs = {
    "add_generation_prompt": True,
    "enable_thinking": False,
}

# Create prompt cache directly
prompt_cache = make_prompt_cache(model.model)

# Generate the initial tool call
llm_response = model.chat(
    messages,
    max_tokens=2048,
    verbose=True,
    tools=tools,
    prompt_cache=prompt_cache,
)
response = llm_response["content"]
logger.gray("Response:")
logger.success(response)

# Parse the tool call

tool_call = parse_tool_call(response)
logger.debug(f"tool_call arguments: {tool_call['arguments']}")
logger.debug(
    f"Type of a: {type(tool_call['arguments']['a'])}, Value: {tool_call['arguments']['a']}")
logger.debug(
    f"Type of b: {type(tool_call['arguments']['b'])}, Value: {tool_call['arguments']['b']}")
tool_result = base_tools[tool_call["name"]](**tool_call["arguments"])

logger.success(f"tool_result: {tool_result}")

# Put the tool result in the prompt with explicit instruction
messages = [
    {"role": "system", "content": "Confirm the result of the multiplication in a clear sentence."},
    {"role": "user", "content": "Multiply 12234585 and 48838483920."},
    {"role": "tool", "name": tool_call["name"], "content": str(tool_result)},
]

# Create a new prompt cache
# prompt_cache = make_prompt_cache(model.model)

# Generate the final response
llm_response = model.chat(
    messages,
    max_tokens=2048,
    verbose=True,
    # prompt_cache=prompt_cache,
)
response = llm_response["content"]
logger.gray("Response 2:")
logger.success(response)
