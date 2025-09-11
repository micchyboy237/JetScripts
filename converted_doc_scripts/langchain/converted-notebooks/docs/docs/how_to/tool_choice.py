from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.tools import tool
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
# How to force models to call a tool

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [LangChain Tools](/docs/concepts/tools)
- [How to use a model to call tools](/docs/how_to/tool_calling)
:::

In order to force our LLM to select a specific [tool](/docs/concepts/tools/), we can use the `tool_choice` parameter to ensure certain behavior. First, let's define our model and tools:
"""
logger.info("# How to force models to call a tool")



@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

# from getpass import getpass


# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOllama(model="llama3.2")

"""
For example, we can force our tool to call the multiply tool by using the following code:
"""
logger.info("For example, we can force our tool to call the multiply tool by using the following code:")

llm_forced_to_multiply = llm.bind_tools(tools, tool_choice="multiply")
llm_forced_to_multiply.invoke("what is 2 + 4")

"""
Even if we pass it something that doesn't require multiplcation - it will still call the tool!

We can also just force our tool to select at least one of our tools by passing in the "any" (or "required" [which is Ollama specific](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.BaseChatOllama.html#jet.adapters.langchain.chat_ollama.chat_models.base.BaseChatOllama.bind_tools)) keyword to the `tool_choice` parameter.
"""
logger.info("Even if we pass it something that doesn't require multiplcation - it will still call the tool!")

llm_forced_to_use_tool = llm.bind_tools(tools, tool_choice="any")
llm_forced_to_use_tool.invoke("What day is today?")

logger.info("\n\n[DONE]", bright=True)