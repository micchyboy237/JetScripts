from jet.logger import logger
from langchain_perplexity import ChatPerplexity
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
# Perplexity

>[Perplexity](https://www.perplexity.ai/pro) is the most powerful way to search
> the internet with unlimited Pro Search, upgraded AI models, unlimited file upload,
> image generation, and API credits.
>
> You can check a [list of available models](https://docs.perplexity.ai/docs/model-cards).

## Installation and Setup

Install the Perplexity x LangChain integration package:
"""
logger.info("# Perplexity")

pip install langchain-perplexity

"""
Get your API key from [here](https://docs.perplexity.ai/docs/getting-started).

## Chat models

See a variety of usage examples [here](/docs/integrations/chat/perplexity).
"""
logger.info("## Chat models")


logger.info("\n\n[DONE]", bright=True)