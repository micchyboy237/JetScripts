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
# Neo4j as Graph Memory

## Prerequisites

### 1. Install Mem0 with Graph Memory support

To use Mem0 with Graph Memory support, install it using pip:

```bash
pip install "mem0ai[graph]"
```

This command installs Mem0 along with the necessary dependencies for graph functionality.

### 2. Install Neo4j

To utilize Neo4j as Graph Memory, run it with Docker:

```bash
docker run \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

This command starts Neo4j with default credentials (`neo4j` / `password`) and exposes both the HTTP (7474) and Bolt (7687) ports.

You can access the Neo4j browser at [http://localhost:7474](http://localhost:7474).

Additional information can be found in the [Neo4j documentation](https://neo4j.com/docs/).

## Configuration

Do all the imports and configure MLX (enter your MLX API key):
"""
logger.info("# Neo4j as Graph Memory")



# os.environ["OPENAI_API_KEY"] = ""

"""
Set up configuration to use the embedder model and Neo4j as a graph store:
"""
logger.info("Set up configuration to use the embedder model and Neo4j as a graph store:")

config = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-large", "embedding_dims": 1536},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://54.87.227.131:7687",
            "username": "neo4j",
            "password": "causes-bins-vines",
        },
    },
}

"""
## Graph Memory initializiation

Initialize Neo4j as a Graph Memory store:
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
Store memories in Neo4j:
"""
logger.info("Store memories in Neo4j:")

result = m.add(messages, user_id="alice")

"""
![](https://github.com/tomasonjo/mem0/blob/neo4jexample/examples/graph-db-demo/alice-memories.png?raw=1)

## Search memories
"""
logger.info("## Search memories")

for result in m.search("what does alice love?", user_id="alice")["results"]:
    logger.debug(result["memory"], result["score"])

logger.info("\n\n[DONE]", bright=True)