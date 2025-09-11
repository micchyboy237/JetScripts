from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
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
# Wolfram Alpha

>[WolframAlpha](https://en.wikipedia.org/wiki/WolframAlpha) is an answer engine developed by `Wolfram Research`.
> It answers factual queries by computing answers from externally sourced data.

This page covers how to use the `Wolfram Alpha API` within LangChain.

## Installation and Setup
- Install requirements with
"""
logger.info("# Wolfram Alpha")

pip install wolframalpha

"""
- Go to wolfram alpha and sign up for a developer account [here](https://developer.wolframalpha.com/)
- Create an app and get your `APP ID`
- Set your APP ID as an environment variable `WOLFRAM_ALPHA_APPID`


## Wrappers

### Utility

There exists a WolframAlphaAPIWrapper utility which wraps this API. To import this utility:
"""
logger.info("## Wrappers")


"""
For a more detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/wolfram_alpha).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
"""
logger.info("### Tool")

tools = load_tools(["wolfram-alpha"])

"""
For more information on tools, see [this page](/docs/how_to/tools_builtin).
"""
logger.info("For more information on tools, see [this page](/docs/how_to/tools_builtin).")

logger.info("\n\n[DONE]", bright=True)