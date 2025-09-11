from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_ollama_function
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
# How to convert tools to Ollama Functions

This notebook goes over how to use LangChain [tools](/docs/concepts/tools/) as Ollama functions.
"""
logger.info("# How to convert tools to Ollama Functions")

# %pip install -qU langchain-community langchain-ollama


model = ChatOllama(model="llama3.2")

tools = [MoveFileTool()]
functions = [convert_to_ollama_function(t) for t in tools]

functions[0]

message = model.invoke(
    [HumanMessage(content="move file foo to bar")], functions=functions
)

message

message.additional_kwargs["function_call"]

"""
With Ollama chat models we can also automatically bind and convert function-like objects with `bind_functions`
"""
logger.info("With Ollama chat models we can also automatically bind and convert function-like objects with `bind_functions`")

model_with_functions = model.bind_functions(tools)
model_with_functions.invoke([HumanMessage(content="move file foo to bar")])

"""
Or we can use the update Ollama API that uses `tools` and `tool_choice` instead of `functions` and `function_call` by using `ChatOllama.bind_tools`:
"""
logger.info("Or we can use the update Ollama API that uses `tools` and `tool_choice` instead of `functions` and `function_call` by using `ChatOllama.bind_tools`:")

model_with_tools = model.bind_tools(tools)
model_with_tools.invoke([HumanMessage(content="move file foo to bar")])

logger.info("\n\n[DONE]", bright=True)