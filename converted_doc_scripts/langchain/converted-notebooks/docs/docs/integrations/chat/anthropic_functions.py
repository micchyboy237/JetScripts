from jet.adapters.langchain.chat_ollama.experimental import ChatOllamaTools
from jet.logger import logger
from pydantic import BaseModel
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
---
sidebar_class_name: hidden
---

# [Deprecated] Experimental Ollama Tools Wrapper

:::warning

The Ollama API officially supports tool-calling so this workaround is no longer needed. Please use [ChatOllama](/docs/integrations/chat/anthropic) with `langchain-anthropic>=0.1.15`.

:::

This notebook shows how to use an experimental wrapper around Ollama that gives it tool calling and structured output capabilities. It follows Ollama's guide [here](https://docs.anthropic.com/claude/docs/functions-external-tools)

The wrapper is available from the `langchain-anthropic` package, and it also requires the optional dependency `defusedxml` for parsing XML output from the llm.

Note: this is a beta feature that will be replaced by Ollama's formal implementation of tool calling, but it is useful for testing and experimentation in the meantime.
"""
logger.info("# [Deprecated] Experimental Ollama Tools Wrapper")

# %pip install -qU langchain-anthropic defusedxml

"""
## Tool Binding

`ChatOllamaTools` exposes a `bind_tools` method that allows you to pass in Pydantic models or BaseTools to the llm.
"""
logger.info("## Tool Binding")



class Person(BaseModel):
    name: str
    age: int


model = ChatOllamaTools(model="claude-3-opus-20240229").bind_tools(tools=[Person])
model.invoke("I am a 27 year old named Erick")

"""
## Structured Output

`ChatOllamaTools` also implements the [`with_structured_output` spec](/docs/how_to/structured_output) for extracting values. Note: this may not be as stable as with models that explicitly offer tool calling.
"""
logger.info("## Structured Output")

chain = ChatOllamaTools(model="claude-3-opus-20240229").with_structured_output(
    Person
)
chain.invoke("I am a 27 year old named Erick")

logger.info("\n\n[DONE]", bright=True)