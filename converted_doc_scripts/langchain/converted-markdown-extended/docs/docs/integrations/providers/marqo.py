from jet.logger import logger
from langchain_community.vectorstores import Marqo
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
# Marqo

This page covers how to use the Marqo ecosystem within LangChain.

### **What is Marqo?**

Marqo is a tensor search engine that uses embeddings stored in in-memory HNSW indexes to achieve cutting edge search speeds. Marqo can scale to hundred-million document indexes with horizontal index sharding and allows for async and non-blocking data upload and search. Marqo uses the latest machine learning models from PyTorch, Huggingface, Ollama and more. You can start with a pre-configured model or bring your own. The built in ONNX support and conversion allows for faster inference and higher throughput on both CPU and GPU.

Because Marqo include its own inference your documents can have a mix of text and images, you can bring Marqo indexes with data from your other systems into the langchain ecosystem without having to worry about your embeddings being compatible.

Deployment of Marqo is flexible, you can get started yourself with our docker image or [contact us about our managed cloud offering!](https://www.marqo.ai/pricing)

To run Marqo locally with our docker image, [see our getting started.](https://docs.marqo.ai/latest/)

## Installation and Setup

- Install the Python SDK with `pip install marqo`

## Wrappers

### VectorStore

There exists a wrapper around Marqo indexes, allowing you to use them within the vectorstore framework. Marqo lets you select from a range of models for generating embeddings and exposes some preprocessing configurations.

The Marqo vectorstore can also work with existing multimodal indexes where your documents have a mix of images and text, for more information refer to [our documentation](https://docs.marqo.ai/latest/#multi-modal-and-cross-modal-search). Note that instantiating the Marqo vectorstore with an existing multimodal index will disable the ability to add any new documents to it via the langchain vectorstore `add_texts` method.

To import this vectorstore:
"""
logger.info("# Marqo")


"""
For a more detailed walkthrough of the Marqo wrapper and some of its unique features, see [this notebook](/docs/integrations/vectorstores/marqo)
"""
logger.info("For a more detailed walkthrough of the Marqo wrapper and some of its unique features, see [this notebook](/docs/integrations/vectorstores/marqo)")

logger.info("\n\n[DONE]", bright=True)