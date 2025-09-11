from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.tools import AIPluginTool
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

# ChatGPT Plugins

:::warning Deprecated

Ollama has [deprecated plugins](https://ollama.com/index/chatgpt-plugins/).

:::

This example shows how to use ChatGPT Plugins within LangChain abstractions.

Note 1: This currently only works for plugins with no auth.

Note 2: There are almost certainly other ways to do this, this is just a first pass. If you have better ideas, please open a PR!
"""
logger.info("# ChatGPT Plugins")

# %pip install --upgrade --quiet langchain-community



tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")

llm = ChatOllama(model="llama3.2")
tools = load_tools(["requests_all"])
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent_chain.run("what t shirts are available in klarna?")

logger.info("\n\n[DONE]", bright=True)