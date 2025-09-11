from jet.logger import logger
from langchain_community.chat_models import JinaChat
from langchain_community.document_compressors import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
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
# Jina AI

>[Jina AI](https://jina.ai/about-us) is a search AI company. `Jina` helps businesses and developers unlock multimodal data with a better search.

:::caution
For proper compatibility, please ensure you are using the `ollama` SDK at version **0.x**.
:::

## Installation and Setup
- Get a Jina AI API token from [here](https://jina.ai/embeddings/) and set it as an environment variable (`JINA_API_TOKEN`)

## Chat Models
"""
logger.info("# Jina AI")


"""
See a [usage examples](/docs/integrations/chat/jinachat).

## Embedding Models

You can check the list of available models from [here](https://jina.ai/embeddings/)
"""
logger.info("## Embedding Models")


"""
See a [usage examples](/docs/integrations/text_embedding/jina).

## Document Transformers

### Jina Rerank
"""
logger.info("## Document Transformers")


"""
See a [usage examples](/docs/integrations/document_transformers/jina_rerank).
"""
logger.info("See a [usage examples](/docs/integrations/document_transformers/jina_rerank).")

logger.info("\n\n[DONE]", bright=True)