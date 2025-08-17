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
[Weaviate](https://weaviate.io/) is an open-source vector search engine. It allows efficient storage and retrieval of high-dimensional vector embeddings, enabling powerful search and retrieval capabilities.


### Installation
"""
logger.info("### Installation")

pip install weaviate weaviate-client

"""
### Usage
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "weaviate",
        "config": {
            "collection_name": "test",
            "cluster_url": "http://localhost:8080",
            "auth_client_secret": None,
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movie? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Config

Let's see the available parameters for the `weaviate` config:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `collection_name` | The name of the collection to store the vectors | `mem0` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `cluster_url` | URL for the Weaviate server | `None` |
| `auth_client_secret` | API key for Weaviate authentication | `None` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)