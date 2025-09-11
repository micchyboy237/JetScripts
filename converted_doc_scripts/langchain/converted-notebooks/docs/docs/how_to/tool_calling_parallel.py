from jet.logger import logger
from langchain.chat_models import init_chat_model
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
# How to disable parallel tool calling

:::info Provider-specific

This API is currently only supported by Ollama and Ollama.

:::

Ollama tool calling performs tool calling in parallel by default. That means that if we ask a question like "What is the weather in Tokyo, New York, and Chicago?" and we have a tool for getting the weather, it will call the tool 3 times in parallel. We can force it to call only a single tool once by using the ``parallel_tool_call`` parameter.

First let's set up our tools and model:
"""
logger.info("# How to disable parallel tool calling")



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

llm = init_chat_model("ollama:gpt-4.1-mini")

"""
Now let's show a quick example of how disabling parallel tool calls work:
"""
logger.info("Now let's show a quick example of how disabling parallel tool calls work:")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
llm_with_tools.invoke("Please call the first tool two times").tool_calls

"""
As we can see, even though we explicitly told the model to call a tool twice, by disabling parallel tool calls the model was constrained to only calling one.


"""
logger.info("As we can see, even though we explicitly told the model to call a tool twice, by disabling parallel tool calls the model was constrained to only calling one.")

logger.info("\n\n[DONE]", bright=True)