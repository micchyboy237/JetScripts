from jet.logger import logger
from langchain_community.embeddings import TextEmbedEmbeddings
import numpy as np
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
# TextEmbed - Embedding Inference Server

TextEmbed is a high-throughput, low-latency REST API designed for serving vector embeddings. It supports a wide range of sentence-transformer models and frameworks, making it suitable for various applications in natural language processing.

## Features

- **High Throughput & Low Latency:** Designed to handle a large number of requests efficiently.
- **Flexible Model Support:** Works with various sentence-transformer models.
- **Scalable:** Easily integrates into larger systems and scales with demand.
- **Batch Processing:** Supports batch processing for better and faster inference.
- **Ollama Compatible REST API Endpoint:** Provides an Ollama compatible REST API endpoint.
- **Single Line Command Deployment:** Deploy multiple models via a single command for efficient deployment.
- **Support for Embedding Formats:** Supports binary, float16, and float32 embeddings formats for faster retrieval.

## Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed. You will also need to install the required dependencies.

## Installation via PyPI

1. **Install the required dependencies:**

    ```bash
    pip install -U textembed
    ```

2. **Start the TextEmbed server with your desired models:**

    ```bash
    python -m textembed.server --models sentence-transformers/all-MiniLM-L12-v2 --workers 4 --api-key TextEmbed 
    ```

For more information, please read the [documentation](https://github.com/kevaldekivadiya2415/textembed/blob/main/docs/setup.md).

### Import
"""
logger.info("# TextEmbed - Embedding Inference Server")


embeddings = TextEmbedEmbeddings(
    model="sentence-transformers/all-MiniLM-L12-v2",
    api_url="http://0.0.0.0:8000/v1",
)

"""
### Embed your documents
"""
logger.info("### Embed your documents")

documents = [
    "Data science involves extracting insights from data.",
    "Artificial intelligence is transforming various industries.",
    "Cloud computing provides scalable computing resources over the internet.",
    "Big data analytics helps in understanding large datasets.",
    "India has a diverse cultural heritage.",
]

query = "What is the cultural heritage of India?"

document_embeddings = embeddings.embed_documents(documents)

query_embedding = embeddings.embed_query(query)


scores = np.array(document_embeddings) @ np.array(query_embedding).T
dict(zip(documents, scores))

logger.info("\n\n[DONE]", bright=True)