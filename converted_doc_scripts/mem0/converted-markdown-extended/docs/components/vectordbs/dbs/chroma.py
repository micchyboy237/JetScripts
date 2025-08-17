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
[Chroma](https://www.trychroma.com/) is an AI-native open-source vector database that simplifies building LLM apps by providing tools for storing, embedding, and searching embeddings with a focus on simplicity and speed.

### Usage
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Config

Here are the parameters available for configuring Chroma:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `collection_name` | The name of the collection | `mem0` |
| `client` | Custom client for Chroma | `None` |
| `path` | Path for the Chroma database | `db` |
| `host` | The host where the Chroma server is running | `None` |
| `port` | The port where the Chroma server is running | `None` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)