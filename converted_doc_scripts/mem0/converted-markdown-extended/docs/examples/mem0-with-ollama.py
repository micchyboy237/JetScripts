from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Mem0 with Ollama
---

## Running Mem0 Locally with Ollama

Mem0 can be utilized entirely locally by leveraging Ollama for both the embedding model and the language model (LLM). This guide will walk you through the necessary steps and provide the complete code to get you started.

### Overview

By using Ollama, you can run Mem0 locally, which allows for greater control over your data and models. This setup uses Ollama for both the embedding model and the language model, providing a fully local solution.

### Setup

Before you begin, ensure you have Mem0 and Ollama installed and properly configured on your local machine.

### Full Code Example

Below is the complete code to set up and use Mem0 locally with Ollama:
"""
logger.info("## Running Mem0 Locally with Ollama")


config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,  # Change this according to your local model's dimensions
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

m = Memory.from_config(config)

m.add("I'm visiting Paris", user_id="john")

memories = m.get_all(user_id="john")

"""
### Key Points

- **Configuration**: The setup involves configuring the vector store, language model, and embedding model to use local resources.
- **Vector Store**: Qdrant is used as the vector store, running on localhost.
- **Language Model**: Ollama is used as the LLM provider, with the "llama3.1:latest" model.
- **Embedding Model**: Ollama is also used for embeddings, with the "nomic-embed-text:latest" model.

### Conclusion

This local setup of Mem0 using Ollama provides a fully self-contained solution for memory management and AI interactions. It allows for greater control over your data and models while still leveraging the powerful capabilities of Mem0.
"""
logger.info("### Key Points")

logger.info("\n\n[DONE]", bright=True)