from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import StackExchangeAPIWrapper
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
# Stack Exchange

>[Stack Exchange](https://en.wikipedia.org/wiki/Stack_Exchange) is a network of
question-and-answer (Q&A) websites on topics in diverse fields, each site covering
a specific topic, where questions, answers, and users are subject to a reputation award process.

This page covers how to use the `Stack Exchange API` within LangChain.

## Installation and Setup
- Install requirements with
"""
logger.info("# Stack Exchange")

pip install stackapi

"""
## Wrappers

### Utility

There exists a StackExchangeAPIWrapper utility which wraps this API. To import this utility:
"""
logger.info("## Wrappers")


"""
For a more detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/stackexchange).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
"""
logger.info("### Tool")

tools = load_tools(["stackexchange"])

"""
For more information on tools, see [this page](/docs/how_to/tools_builtin).
"""
logger.info("For more information on tools, see [this page](/docs/how_to/tools_builtin).")

logger.info("\n\n[DONE]", bright=True)