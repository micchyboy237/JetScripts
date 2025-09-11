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
# MCP Toolbox

The [MCP Toolbox](https://googleapis.github.io/genai-toolbox/getting-started/introduction/) in LangChain allows you to equip an agent with a set of tools. When the agent receives a query, it can intelligently select and use the most appropriate tool provided by MCP Toolbox to fulfill the request.

## What is it?

MCP Toolbox is essentially a container for your tools. Think of it as a multi-tool device for your agent; it can hold any tools you create. The agent then decides which specific tool to use based on the user's input.

This is particularly useful when you have an agent that needs to perform a variety of tasks that require different capabilities.

## Installation

To get started, you'll need to install the necessary package:
"""
logger.info("# MCP Toolbox")

pip install toolbox-langchain

"""
## Tutorial

For a complete, step-by-step guide on how to create, configure, and use MCP Toolbox with your agents, please refer to our detailed Jupyter notebook tutorial.

**[➡️ View the full tutorial here](/docs/integrations/tools/toolbox)**.
"""
logger.info("## Tutorial")

logger.info("\n\n[DONE]", bright=True)