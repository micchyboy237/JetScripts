from jet.logger import logger
from langchain_core.documents import Document
from langchain_nebius import ChatNebius
from langchain_nebius import NebiusEmbeddings
from langchain_nebius import NebiusEmbeddings, NebiusRetriever
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, NebiusRetrievalTool
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
# Nebius

All functionality related to Nebius AI Studio

>[Nebius AI Studio](https://studio.nebius.ai/) provides API access to a wide range of state-of-the-art large language models and embedding models for various use cases.

## Installation and Setup

The Nebius integration can be installed via pip:
"""
logger.info("# Nebius")

pip install langchain-nebius

"""
To use Nebius AI Studio, you'll need an API key which you can obtain from [Nebius AI Studio](https://studio.nebius.ai/). The API key can be passed as an initialization parameter `api_key` or set as the environment variable `NEBIUS_API_KEY`.
"""
logger.info("To use Nebius AI Studio, you'll need an API key which you can obtain from [Nebius AI Studio](https://studio.nebius.ai/). The API key can be passed as an initialization parameter `api_key` or set as the environment variable `NEBIUS_API_KEY`.")

os.environ["NEBIUS_API_KEY"] = "YOUR-NEBIUS-API-KEY"

"""
### Available Models

The full list of supported models can be found in the [Nebius AI Studio Documentation](https://studio.nebius.com/).


## Chat models

### ChatNebius

The `ChatNebius` class allows you to interact with Nebius AI Studio's chat models.

See a [usage example](/docs/integrations/chat/nebius).
"""
logger.info("### Available Models")


chat = ChatNebius(
    model="Qwen/Qwen3-30B-A3B-fast",  # Choose from available models
    temperature=0.6,
    top_p=0.95
)

"""
## Embedding models

### NebiusEmbeddings

The `NebiusEmbeddings` class allows you to generate vector embeddings using Nebius AI Studio's embedding models.

See a [usage example](/docs/integrations/text_embedding/nebius).
"""
logger.info("## Embedding models")


embeddings = NebiusEmbeddings(
    model="BAAI/bge-en-icl"  # Default embedding model
)

"""
## Retrievers

### NebiusRetriever

The `NebiusRetriever` enables efficient similarity search using embeddings from Nebius AI Studio. It leverages high-quality embedding models to enable semantic search over documents.

See a [usage example](/docs/integrations/retrievers/nebius).
"""
logger.info("## Retrievers")


docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
]

embeddings = NebiusEmbeddings()

retriever = NebiusRetriever(
    embeddings=embeddings,
    docs=docs,
    k=2  # Number of documents to return
)

"""
## Tools

### NebiusRetrievalTool

The `NebiusRetrievalTool` allows you to create a tool for agents based on the NebiusRetriever.
"""
logger.info("## Tools")


docs = [
    Document(page_content="Paris is the capital of France and has the Eiffel Tower"),
    Document(page_content="Berlin is the capital of Germany and has the Brandenburg Gate"),
]

embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs)

tool = NebiusRetrievalTool(
    retriever=retriever,
    name="nebius_search",
    description="Search for information about European capitals"
)

logger.info("\n\n[DONE]", bright=True)