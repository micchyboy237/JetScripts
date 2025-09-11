from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# How to bind model-specific tools

Providers adopt different conventions for formatting tool schemas. 
For instance, Ollama uses a format like this:

- `type`: The type of the tool. At the time of writing, this is always `"function"`.
- `function`: An object containing tool parameters.
- `function.name`: The name of the schema to output.
- `function.description`: A high level description of the schema to output.
- `function.parameters`: The nested details of the schema you want to extract, formatted as a [JSON schema](https://json-schema.org/) dict.

We can bind this model-specific format directly to the model as well if preferred. Here's an example:
"""
logger.info("# How to bind model-specific tools")


model = ChatOllama(model="llama3.2")

model_with_tools = model.bind(
    tools=[
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers together.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First integer"},
                        "b": {"type": "number", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
)

model_with_tools.invoke("Whats 119 times 8?")

"""
This is functionally equivalent to the `bind_tools()` method.
"""
logger.info("This is functionally equivalent to the `bind_tools()` method.")

logger.info("\n\n[DONE]", bright=True)