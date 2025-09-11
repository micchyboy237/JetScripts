from jet.logger import logger
from langchain_community.llms import Arcee
from langchain_community.retrievers import ArceeRetriever
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
# Arcee

>[Arcee](https://www.arcee.ai/about/about-us) enables the development and advancement
> of what we coin as SLMs—small, specialized, secure, and scalable language models.
> By offering a SLM Adaptation System and a seamless, secure integration,
> `Arcee` empowers enterprises to harness the full potential of
> domain-adapted language models, driving the transformative
> innovation in operations.


## Installation and Setup

Get your `Arcee API` key.


## LLMs

See a [usage example](/docs/integrations/llms/arcee).
"""
logger.info("# Arcee")


"""
## Retrievers

See a [usage example](/docs/integrations/retrievers/arcee).
"""
logger.info("## Retrievers")


logger.info("\n\n[DONE]", bright=True)