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
# Tavily

[Tavily](https://tavily.com) is a search engine, specifically designed for AI agents.
Tavily provides both a search and extract API, AI developers can effortlessly integrate their
applications with realtime online information. Tavily’s primary mission is to provide factual
and reliable information from trusted sources, enhancing the accuracy and reliability of AI
generated content and reasoning.

## Installation and Setup
"""
logger.info("# Tavily")

pip install langchain-tavily

"""
## Tools

See detail on available tools [tavily_search](/docs/integrations/tools/tavily_search) and [tavily_extract](/docs/integrations/tools/tavily_extract).
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)