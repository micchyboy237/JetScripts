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
# Memgraph as Graph Memory

## Prerequisites

### 1. Install Mem0 with Graph Memory support 

To use Mem0 with Graph Memory support, install it using pip:

```bash
pip install "mem0ai[graph]"
```

This command installs Mem0 along with the necessary dependencies for graph functionality.

### 2. Install Memgraph

To utilize Memgraph as Graph Memory, run it with Docker:

```bash
docker run -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=True
```

The `--schema-info-enabled` flag is set to `True` for more performant schema
generation.

Additional information can be found on [Memgraph documentation](https://memgraph.com/docs).

## Configuration

Do all the imports and configure MLX (enter your MLX API key):
"""
logger.info("# Memgraph as Graph Memory")



# os.environ["OPENAI_API_KEY"] = ""

"""
Set up configuration to use the embedder model and Memgraph as a graph store:
"""
logger.info("Set up configuration to use the embedder model and Memgraph as a graph store:")

config = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-large", "embedding_dims": 1536},
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "mem0graph",
        },
    },
}

"""
## Graph Memory initializiation 

Initialize Memgraph as a Graph Memory store:
"""
logger.info("## Graph Memory initializiation")

m = Memory.from_config(config_dict=config)

"""
## Store memories 

Create memories:
"""
logger.info("## Store memories")

messages = [
    {
        "role": "user",
        "content": "I'm planning to watch a movie tonight. Any recommendations?",
    },
    {
        "role": "assistant",
        "content": "How about a thriller movies? They can be quite engaging.",
    },
    {
        "role": "user",
        "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
    },
    {
        "role": "assistant",
        "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
    },
]

"""
Store memories in Memgraph:
"""
logger.info("Store memories in Memgraph:")

result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"})

"""
![](./alice-memories.png)

## Search memories
"""
logger.info("## Search memories")

for result in m.search("what does alice love?", user_id="alice")["results"]:
    logger.debug(result["memory"], result["score"])

logger.info("\n\n[DONE]", bright=True)