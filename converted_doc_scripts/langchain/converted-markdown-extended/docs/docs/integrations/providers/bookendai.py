from jet.logger import logger
from langchain_community.embeddings import BookendEmbeddings
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
# bookend.ai

LangChain implements an integration with embeddings provided by [bookend.ai](https://bookend.ai/).


## Installation and Setup


You need to register and get the `API_KEY`
from the [bookend.ai](https://bookend.ai/) website.

## Embedding model

See a [usage example](/docs/integrations/text_embedding/bookend).
"""
logger.info("# bookend.ai")


logger.info("\n\n[DONE]", bright=True)