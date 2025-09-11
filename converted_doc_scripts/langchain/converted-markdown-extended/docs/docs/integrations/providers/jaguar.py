from jet.logger import logger
from langchain_community.vectorstores.jaguar import Jaguar
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
# Jaguar

This page describes how to use Jaguar vector database within LangChain.
It contains three sections: introduction, installation and setup, and Jaguar API.


## Introduction

Jaguar vector database has the following characteristics:

1. It is a distributed vector database
2. The “ZeroMove” feature of JaguarDB enables instant horizontal scalability
3. Multimodal: embeddings, text, images, videos, PDFs, audio, time series, and geospatial
4. All-masters: allows both parallel reads and writes
5. Anomaly detection capabilities
6. RAG support: combines LLM with proprietary and real-time data
7. Shared metadata: sharing of metadata across multiple vector indexes
8. Distance metrics: Euclidean, Cosine, InnerProduct, Manhatten, Chebyshev, Hamming, Jeccard, Minkowski

[Overview of Jaguar scalable vector database](http://www.jaguardb.com)

You can run JaguarDB in docker container; or download the software and run on-cloud or off-cloud.

## Installation and Setup

- Install the JaguarDB on one host or multiple hosts
- Install the Jaguar HTTP Gateway server on one host
- Install the JaguarDB HTTP Client package

The steps are described in [Jaguar Documents](http://www.jaguardb.com/support.html)

Environment Variables in client programs:

#     export OPENAI_API_KEY="......"
    export JAGUAR_API_KEY="......"


## Jaguar API

Together with LangChain, a Jaguar client class is provided by importing it in Python:
"""
logger.info("# Jaguar")


"""
Supported API functions of the Jaguar class are:

- `add_texts`
- `add_documents`
- `from_texts`
- `from_documents`
- `similarity_search`
- `is_anomalous`
- `create`
- `delete`
- `clear`
- `drop`
- `login`
- `logout`


For more details of the Jaguar API, please refer to [this notebook](/docs/integrations/vectorstores/jaguar)
"""
logger.info("Supported API functions of the Jaguar class are:")

logger.info("\n\n[DONE]", bright=True)