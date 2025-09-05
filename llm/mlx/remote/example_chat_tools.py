from jet.llm.mlx.remote import generation as gen
from typing import List, Dict
from jet.llm.mlx.remote.types import Message

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


def llama_tool_example(query: str) -> None:
    """Demonstrate tool usage with Llama-3.2-3B-Instruct-4bit model."""
    print("=== Llama Chat Completion with Tools ===")

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

    messages: List[Message] = [
        {"role": "user", "content": query, "tool_calls": []}
    ]
    response = gen.chat(
        messages=messages,
        system_prompt=SYSTEM_PROMPT,
        model="mlx-community/llama-3.2-3b-instruct-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    print("Response:")
    print(f"Content: {response.get('content', 'No content')}")
    print(f"Tool Calls: {response.get('tool_calls', 'No tool calls')}")
    for choice in response["choices"]:
        print(f"Choice {choice['index']}: {choice['message']}")


def mistral_tool_example(query: str) -> None:
    """Demonstrate tool usage with Mistral-7B-Instruct-v0.3-4bit model."""
    print("=== Mistral Chat Completion with Tools ===")

    def calculate(expression: str) -> float:
        """
        Perform a mathematical calculation.

        Args:
            expression (str): The mathematical expression to evaluate.

        Returns:
            float: The result of the calculation.
        """
        try:
            # WARNING: Using eval is dangerous in production!
            # For demo purposes only. In production, use a safe math parser.
            return eval(expression, {"__builtins__": {}})
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}") from e

    tools = [calculate]
    messages: List[Message] = [
        {"role": "user", "content": query, "tool_calls": []}
    ]
    response = gen.chat(
        messages=messages,
        system_prompt=SYSTEM_PROMPT,
        model="mlx-community/mistral-7b-instruct-v0.3-4bit",
        tools=tools,
        max_tokens=100,
        verbose=True,
    )
    print("Response:")
    print(f"Content: {response.get('content', 'No content')}")
    print(f"Tool Calls: {response.get('tool_calls', 'No tool calls')}")
    for choice in response["choices"]:
        print(f"Choice {choice['index']}: {choice['message']}")


def main():
    query = "What is three thousand four hundred twenty three plus 6 thousand nine hundred ninety nine?"

    print("=== Chat Completion Examples with Tools ===")
    llama_tool_example(query)
    print("\n" + "="*50 + "\n")
    mistral_tool_example(query)


if __name__ == "__main__":
    main()
