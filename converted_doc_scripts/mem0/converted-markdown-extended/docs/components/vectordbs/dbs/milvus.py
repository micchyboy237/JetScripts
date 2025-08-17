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
[Milvus](https://milvus.io/) Milvus is an open-source vector database that suits AI applications of every size from running a demo chatbot in Jupyter notebook to building web-scale search that serves billions of users.

### Usage
"""
logger.info("### Usage")


config = {
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "test",
            "embedding_model_dims": "123",
            "url": "127.0.0.1",
            "token": "8e4b8ca8cf2c67",
            "db_name": "my_database",
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

Here's the parameters available for configuring Milvus Database:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `url` | Full URL/Uri for Milvus/Zilliz server | `http://localhost:19530` |
| `token` | Token for Zilliz server / for local setup defaults to None. | `None` |
| `collection_name` | The name of the collection | `mem0` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `metric_type` | Metric type for similarity search | `L2` |
| `db_name` | Name of the database | `""` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)