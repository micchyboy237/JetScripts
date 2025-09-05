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

# Updated system prompt to handle any argument type and query flexibly
SYSTEM_PROMPT = """
You are a helpful assistant capable of using tools to provide accurate and contextually appropriate responses. ALL responses MUST be returned as a JSON array containing a single object with the exact schema:
[
  {
    "function": {
      "name": "<tool_name>",
      "arguments": {
        "<argument_name>": "<argument_value>"
        ...
      }
    }
  }
]
Ensure the response is a valid JSON array, using the keys exactly as specified. If no tool is needed, return an empty array []. Follow the user's query precisely. When processing tool arguments, strictly adhere to the types specified in the tool's signature if provided (e.g., int, str, float, bool, list). If types are not explicitly defined, infer sensible types based on the query context (e.g., numerical inputs as int or float, text as str, collections as list or dict). For numerical queries, accurately interpret natural language numbers. For complex queries, dynamically adapt to the input structure and content, ensuring flexibility for diverse argument types and query formats. Maintain clarity, precision, and robustness in all responses, using reasonable defaults for unspecified arguments (e.g., temperature unit based on location or context).
"""


def main(query: str, model: LLMModelType):
    model_id = resolve_model_value(model)
    _model, tokenizer = load(model_id)

    def add_two_numbers(a: int, b: int) -> int:
        """
        Add two numbers

        Args:
            a (int): The first number
            b (int): The second number

        Returns:
            int: The sum of the two numbers
        """
        return a + b

    tools = [add_two_numbers]

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
            "content": f"{SYSTEM_PROMPT}\n\n{tool_description}\n\n{query}"
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
    # Ensure output is a JSON array
    try:
        output_dict = json.loads(output)
        if not isinstance(output_dict, list):
            output_dict = [output_dict] if output_dict else []
    except json.JSONDecodeError:
        output_dict = []

    filename = format_sub_dir(resolve_model_key(model))

    print(f"Model: {model_id}")
    print(f"Response:\n{format_json(output_dict)}")

    # Execute tool call if present in the response
    tool_result = None
    if output_dict and isinstance(output_dict, list) and len(output_dict) > 0:
        tool_call = output_dict[0].get("function", {})
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        for tool in tools:
            if tool.__name__ == tool_name:
                try:
                    tool_result = tool(**tool_args)
                    print(
                        f"Tool {tool_name} executed with result: {tool_result}")
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")

    save_file({
        "prompt": tokenizer.decode(prompt),
        "response": output_dict,
        "tool_result": tool_result
    }, f"{OUTPUT_DIR}/{filename}_tool_usage.json")

    # Save output as markdown
    markdown_content = f"""# Tool Usage Report

## Model
{model_id}

## Prompt
{tokenizer.decode(prompt)}

## Response
```json
{format_json(output_dict)}
```

"""
    if tool_result is not None:
        markdown_content += f"## Tool Result\n{tool_result}\n"

    print(f"Model: {model_id}")
    print(f"Response:\n{format_json(output_dict)}")
    save_file(
        markdown_content,
        f"{OUTPUT_DIR}/{filename}_tool_usage.md",
    )


if __name__ == "__main__":
    query = "What is three thousand four hundred twenty three plus 6 thousand nine hundred ninety nine?"
    main(query, "mlx-community/Llama-3.2-3B-Instruct-4bit")
    main(query, "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
