import os
import shutil
import json
import mlx.core as mx
from mlx_lm import load, generate
from jet.llm.mlx.mlx_utils import has_tools
from jet.models.model_types import LLMModelType
from jet.models.utils import resolve_model_key, resolve_model_value
from jet.file.utils import save_file
from jet.utils.text import format_sub_dir
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import get_method_info

# Reset MLX cache
mx.clear_cache()

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Existing imports and code remain unchanged up to the system_prompt
SYSTEM_PROMPT = """
You are a helpful assistant capable of using tools to provide accurate responses. When responding to a query that requires tool usage, you MUST return a single JSON object with the exact schema:
{
  "name": "<tool_name>",
  "arguments": {
    "<argument_name>": "<argument_value>",
    ...
  }
}
Ensure the response is a single valid JSON, uses the keys exactly as specified. If no tool is needed, respond with plain text. Follow the user's query precisely and infer reasonable defaults for unspecified arguments (e.g., temperature unit based on location). Maintain clarity and precision in all responses.
"""


def main(query: str, model: LLMModelType):
    model_id = resolve_model_value(model)
    _model, tokenizer = load(model_id)

    def get_current_weather(location: str, format: str):
        """
        Retrieves the current weather for a specified location

        Args:
            location (str): The city and state, e.g., San Francisco, CA
            format (str): The temperature unit to use, either 'celsius' or 'fahrenheit'

        Returns:
            None: This function is a placeholder and does not return a value
        """
        pass

    tools = [get_current_weather]

    # Add tool description to the user prompt
    if has_tools(model):
        tool_description = "Available tools:\n"
        for tool in tools:
            tool_info = get_method_info(tool)
            tool_description += format_json(tool_info) + "\n"
        tool_description = tool_description.rstrip()  # Remove trailing newline
    else:
        tool_description = ""

    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\n{tool_description}\n{query}"
        }
    ]

    # Format and tokenize the tool use prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        add_special_tokens=False,
    )

    # Generate response
    output = generate(
        _model,
        tokenizer,
        prompt=prompt,
        max_tokens=1000,
        verbose=True,
    )
    output_dict = json.loads(output)

    filename = format_sub_dir(resolve_model_key(model))

    print(f"Model: {model_id}")
    print(f"Response:\n{format_json(output_dict)}")
    save_file({
        "prompt": messages,
        "response": output_dict,
    }, f"{OUTPUT_DIR}/{filename}_tool_usage.json")


if __name__ == "__main__":
    query = "What's the weather like in Paris?"
    main(query, "mlx-community/Llama-3.2-3B-Instruct-4bit")
    main(query, "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
