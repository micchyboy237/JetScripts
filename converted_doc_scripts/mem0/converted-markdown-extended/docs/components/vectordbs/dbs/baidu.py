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
title: Baidu VectorDB (Mochow)
---

[Baidu VectorDB](https://cloud.baidu.com/doc/VDB/index.html) is an enterprise-level distributed vector database service developed by Baidu Intelligent Cloud. It is powered by Baidu's proprietary "Mochow" vector database kernel, providing high performance, availability, and security for vector search.

### Usage
"""
logger.info("### Usage")


config = {
    "vector_store": {
        "provider": "baidu",
        "config": {
            "endpoint": "http://your-mochow-endpoint:8287",
            "account": "root",
            "api_key": "your-api-key",
            "database_name": "mem0",
            "table_name": "mem0_table",
            "embedding_model_dims": 1536,
            "metric_type": "COSINE"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movie? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Config

Here are the available parameters for the `mochow` config:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `endpoint` | Endpoint URL for your Baidu VectorDB instance | Required |
| `account` | Baidu VectorDB account name | `root` |
| `api_key` | API key for accessing Baidu VectorDB | Required |
| `database_name` | Name of the database | `mem0` |
| `table_name` | Name of the table | `mem0_table` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `metric_type` | Distance metric for similarity search | `L2` |

### Distance Metrics

The following distance metrics are supported:

- `L2`: Euclidean distance (default)
- `IP`: Inner product
- `COSINE`: Cosine similarity

### Index Configuration

The vector index is automatically configured with the following HNSW parameters:

- `m`: 16 (number of connections per element)
- `efconstruction`: 200 (size of the dynamic candidate list)
- `auto_build`: true (automatically build index)
- `auto_build_index_policy`: Incremental build with 10000 rows increment
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)