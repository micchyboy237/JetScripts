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
title: Hugging Face
---

You can use embedding models from Huggingface to run Mem0 locally.

### Usage
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM

config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="john")

"""
### Using Text Embeddings Inference (TEI)

You can also use Hugging Face's Text Embeddings Inference service for faster and more efficient embeddings:
"""
logger.info("### Using Text Embeddings Inference (TEI)")


# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM

config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "huggingface_base_url": "http://localhost:3000/v1"
        }
    }
}

m = Memory.from_config(config)
m.add("This text will be embedded using the TEI service.", user_id="john")

"""
To run the TEI service, you can use Docker:
"""
logger.info("To run the TEI service, you can use Docker:")

docker run -d -p 3000:80 -v huggingfacetei:/data --platform linux/amd64 \
    ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 \
    --model-id BAAI/bge-small-en-v1.5

"""
### Config

Here are the parameters available for configuring Huggingface embedder:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the model to use | `multi-qa-MiniLM-L6-cos-v1` |
| `embedding_dims` | Dimensions of the embedding model | `selected_model_dimensions` |
| `model_kwargs` | Additional arguments for the model | `None` |
| `huggingface_base_url` | URL to connect to Text Embeddings Inference (TEI) API | `None` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)