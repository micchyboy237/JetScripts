import os
import shutil
from jet.transformers.formatters import format_json
from mlx_lm import load, generate
from jet.utils.eval_utils import parse_and_evaluate
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

model_id = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(model_id)


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass


conversation = [
    {"role": "user", "content": "What's the weather like in Paris?"}
]
tools = [get_current_weather]

# Format and tokenize the tool use prompt
prompt = tokenizer.apply_chat_template(
    conversation,
    tools=tools,
    add_generation_prompt=True
)

# Generate response
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1000,
    verbose=True,
)

tool_calls = parse_and_evaluate(output)

# Execute tool call if present in the response
tool_result = None
executed_tool_name = None
executed_tool_args = None

if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
    # Expecting format: [{"name": ..., "arguments": {...}}] or [{"function": {"name": ..., "arguments": {...}}}]
    call = tool_calls[0]
    # Support both formats
    if "function" in call:
        tool_call = call["function"]
    else:
        tool_call = call
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("arguments", {})
    for tool in tools:
        if tool.__name__ == tool_name:
            try:
                tool_result = tool(**tool_args)
                executed_tool_name = tool_name
                executed_tool_args = tool_args
                logger.success(
                    f"Tool {tool_name} executed with result: {tool_result}")
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")

logger.success("Tool calls:\n" + format_json(tool_calls))

save_file({
    "prompt": tokenizer.decode(prompt),
    "tool_calls": tool_calls,
    "tool_result": tool_result,
    "executed_tool_name": executed_tool_name,
    "executed_tool_args": executed_tool_args
}, f"{OUTPUT_DIR}/result.json")
