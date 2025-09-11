from jet.logger import logger
from langchain_vectorize import VectorizeRetriever
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
# Vectorize

> [Vectorize](https://vectorize.io/) helps you build AI apps faster and with less hassle.
> It automates data extraction, finds the best vectorization strategy using RAG evaluation,
> and lets you quickly deploy real-time RAG pipelines for your unstructured data.
> Your vector search indexes stay up-to-date, and it integrates with your existing vector database,
> so you maintain full control of your data.
> Vectorize handles the heavy lifting, freeing you to focus on building robust AI solutions without getting bogged down by data management.

# Installation and Setup

Install the following Python package:
"""
logger.info("# Vectorize")

pip install langchain-vectorize

"""
Sign up for a free Vectorize account [here](https://platform.vectorize.io/)
Generate an access token in the [Access Token](https://docs.vectorize.io/rag-pipelines/retrieval-endpoint#access-tokens) section
Gather your organization ID. From the browser url, extract the UUID from the URL after /organization/

Set up the following variables:
"""
logger.info("Sign up for a free Vectorize account [here](https://platform.vectorize.io/)")

VECTORIZE_ORG_ID="your-organization-id"
VECTORIZE_API_TOKEN="your-api-token"

"""
## Retriever
"""
logger.info("## Retriever")


retriever = VectorizeRetriever(
    api_token=VECTORIZE_API_TOKEN,
    organization=VECTORIZE_ORG_ID,
    pipeline_id="...",
)
retriever.invoke("query")

"""
Learn more in the [example notebook](/docs/integrations/retrievers/vectorize).
"""
logger.info("Learn more in the [example notebook](/docs/integrations/retrievers/vectorize).")

logger.info("\n\n[DONE]", bright=True)