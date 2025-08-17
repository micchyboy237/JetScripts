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
# MongoDB

[MongoDB](https://www.mongodb.com/) is a versatile document database that supports vector search capabilities, allowing for efficient high-dimensional similarity searches over large datasets with robust scalability and performance.

## Usage
"""
logger.info("# MongoDB")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "mongodb",
        "config": {
            "db_name": "mem0-db",
            "collection_name": "mem0-collection",
            "mongo_uri":"mongodb://username:password@localhost:27017"
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
## Config

Here are the parameters available for configuring MongoDB:

| Parameter | Description | Default Value |
| --- | --- | --- |
| db_name | Name of the MongoDB database | `"mem0_db"` |
| collection_name | Name of the MongoDB collection | `"mem0_collection"` |
| embedding_model_dims | Dimensions of the embedding vectors | `1536` |
| mongo_uri | The mongo URI connection string | mongodb://username:password@localhost:27017 |

> **Note**: If Mongo_uri is not provided it will default to mongodb://username:password@localhost:27017.
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)