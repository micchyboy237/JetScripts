from jet.logger import logger
from langchain_voyageai import VoyageAIEmbeddings
from langchain_voyageai import VoyageAIRerank
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
# VoyageAI

All functionality related to VoyageAI

>[VoyageAI](https://www.voyageai.com/) Voyage AI builds embedding models, customized for your domain and company, for better retrieval quality.

## Installation and Setup

Install the integration package with
"""
logger.info("# VoyageAI")

pip install langchain-voyageai

"""
Get a VoyageAI API key and set it as an environment variable (`VOYAGE_API_KEY`)


## Text Embedding Model

See a [usage example](/docs/integrations/text_embedding/voyageai)
"""
logger.info("## Text Embedding Model")


"""
## Reranking

See a [usage example](/docs/integrations/document_transformers/voyageai-reranker)
"""
logger.info("## Reranking")


logger.info("\n\n[DONE]", bright=True)