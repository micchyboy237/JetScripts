from dotenv import load_dotenv
from jet.logger import CustomLogger
from mem0 import Memory
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
import boto3
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Neptune as Graph Memory

In this notebook, we will be connecting using a Amazon Neptune Analytics instance as our memory graph storage for Mem0.

The Graph Memory storage persists memories in a graph or relationship form when performing `m.add` memory operations. It then uses vector distance algorithms to find related memories during a `m.search` operation. Relationships are returned in the result, and add context to the memories.

Reference: [Vector Similarity using Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vector-similarity.html)

## Prerequisites

### 1. Install Mem0 with Graph Memory support 

To use Mem0 with Graph Memory support (as well as other Amazon services), use pip install:

```bash
pip install "mem0ai[graph,extras]"
```

This command installs Mem0 along with the necessary dependencies for graph functionality (`graph`) and other Amazon dependencies (`extras`).

### 2. Connect to Amazon services

For this sample notebook, configure `mem0ai` with [Amazon Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) as the graph store, [Amazon OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-overview.html) as the vector store, and [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) for generating embeddings.

Use the following guide for setup details: [Setup AWS Bedrock, AOSS, and Neptune](https://docs.mem0.ai/examples/aws_example#aws-bedrock-and-aoss)

Your configuration should look similar to:

```python
config = {
    "embedder": {
        "provider": "aws_bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v2:0"
        }
    },
    "llm": {
        "provider": "aws_bedrock",
        "config": {
            "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "vector_store": {
        "provider": "opensearch",
        "config": {
            "collection_name": "mem0",
            "host": "your-opensearch-domain.us-west-2.es.amazonaws.com",
            "port": 443,
            "http_auth": auth,
            "connection_class": RequestsHttpConnection,
            "pool_maxsize": 20,
            "use_ssl": True,
            "verify_certs": True,
            "embedding_model_dims": 1024,
        }
    },
    "graph_store": {
        "provider": "neptune",
        "config": {
            "endpoint": f"neptune-graph://my-graph-identifier",
        },
    },
}
```

## Setup

Import all packages and setup logging
"""
logger.info("# Neptune as Graph Memory")


load_dotenv()

logging.getLogger("mem0.graphs.neptune.main").setLevel(logging.DEBUG)
logging.getLogger("mem0.graphs.neptune.base").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    format="%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Explicitly set output to stdout
)

"""
Setup the Mem0 configuration using:
- Amazon Bedrock as the embedder
- Amazon Neptune Analytics instance as a graph store
- OpenSearch as the vector store
"""
logger.info("Setup the Mem0 configuration using:")

bedrock_embedder_model = "amazon.titan-embed-text-v2:0"
bedrock_llm_model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
embedding_model_dims = 1024

graph_identifier = os.environ.get("GRAPH_ID")

opensearch_host = os.environ.get("OS_HOST")
opensearch_post = os.environ.get("OS_PORT")

credentials = boto3.Session().get_credentials()
region = os.environ.get("AWS_REGION")
auth = AWSV4SignerAuth(credentials, region)

config = {
    "embedder": {
        "provider": "aws_bedrock",
        "config": {
            "model": bedrock_embedder_model,
        }
    },
    "llm": {
        "provider": "aws_bedrock",
        "config": {
            "model": bedrock_llm_model,
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "vector_store": {
        "provider": "opensearch",
        "config": {
            "collection_name": "mem0ai_vector_store",
            "host": opensearch_host,
            "port": opensearch_post,
            "http_auth": auth,
            "embedding_model_dims": embedding_model_dims,
            "use_ssl": True,
            "verify_certs": True,
            "connection_class": RequestsHttpConnection,
        },
    },
    "graph_store": {
        "provider": "neptune",
        "config": {
            "endpoint": f"neptune-graph://{graph_identifier}",
        },
    },
}

"""
## Graph Memory initializiation

Initialize Memgraph as a Graph Memory store:
"""
logger.info("## Graph Memory initializiation")

m = Memory.from_config(config_dict=config)

app_id = "movies"
user_id = "alice"

m.delete_all(user_id=user_id)

"""
## Store memories

Create memories and store one at a time:
"""
logger.info("## Store memories")

messages = [
    {
        "role": "user",
        "content": "I'm planning to watch a movie tonight. Any recommendations?",
    },
]

result = m.add(messages, user_id=user_id, metadata={"category": "movie_recommendations"})

all_results = m.get_all(user_id=user_id)
for n in all_results["results"]:
    logger.debug(f"node \"{n['memory']}\": [hash: {n['hash']}]")

for e in all_results["relations"]:
    logger.debug(f"edge \"{e['source']}\" --{e['relationship']}--> \"{e['target']}\"")

"""
## Graph Explorer Visualization

You can visualize the graph using a Graph Explorer connection to Neptune Analytics in Neptune Notebooks in the Amazon console.  See [Using Amazon Neptune with graph notebooks](https://docs.aws.amazon.com/neptune/latest/userguide/graph-notebooks.html) for instructions on how to setup a Neptune Notebook with Graph Explorer.

Once the graph has been generated, you can open the visualization in the Neptune > Notebooks and click on Actions > Open Graph Explorer.  This will automatically connect to your neptune analytics graph that was provided in the notebook setup.

Once in Graph Explorer, visit Open Connections and send all the available nodes and edges to Explorer. Visit Open Graph Explorer to see the nodes and edges in the graph.

### Graph Explorer Visualization Example

_Note that the visualization given below represents only a single example of the possible results generated by the LLM._

Visualization for the relationship:
```
"alice" --plans_to_watch--> "movie"
```

![neptune-example-visualization-1.png](./neptune-example-visualization-1.png)
"""
logger.info("## Graph Explorer Visualization")

messages = [
    {
        "role": "assistant",
        "content": "How about a thriller movies? They can be quite engaging.",
    },
]

result = m.add(messages, user_id=user_id, metadata={"category": "movie_recommendations"})

all_results = m.get_all(user_id=user_id)
for n in all_results["results"]:
    logger.debug(f"node \"{n['memory']}\": [hash: {n['hash']}]")

for e in all_results["relations"]:
    logger.debug(f"edge \"{e['source']}\" --{e['relationship']}--> \"{e['target']}\"")

"""
### Graph Explorer Visualization Example

_Note that the visualization given below represents only a single example of the possible results generated by the LLM._

Visualization for the relationship:
```
"alice" --plans_to_watch--> "movie"
"thriller" --type_of--> "movie"
"movie" --can_be--> "engaging"
```

![neptune-example-visualization-2.png](./neptune-example-visualization-2.png)
"""
logger.info("### Graph Explorer Visualization Example")

messages = [
    {
        "role": "user",
        "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
    },
]

result = m.add(messages, user_id=user_id, metadata={"category": "movie_recommendations"})

all_results = m.get_all(user_id=user_id)
for n in all_results["results"]:
    logger.debug(f"node \"{n['memory']}\": [hash: {n['hash']}]")

for e in all_results["relations"]:
    logger.debug(f"edge \"{e['source']}\" --{e['relationship']}--> \"{e['target']}\"")

"""
### Graph Explorer Visualization Example

_Note that the visualization given below represents only a single example of the possible results generated by the LLM._

Visualization for the relationship:
```
"alice" --dislikes--> "thriller_movies"
"alice" --loves--> "sci-fi_movies"
"alice" --plans_to_watch--> "movie"
"thriller" --type_of--> "movie"
"movie" --can_be--> "engaging"
```

![neptune-example-visualization-3.png](./neptune-example-visualization-3.png)
"""
logger.info("### Graph Explorer Visualization Example")

messages = [
    {
        "role": "assistant",
        "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
    },
]

result = m.add(messages, user_id=user_id, metadata={"category": "movie_recommendations"})

all_results = m.get_all(user_id=user_id)
for n in all_results["results"]:
    logger.debug(f"node \"{n['memory']}\": [hash: {n['hash']}]")

for e in all_results["relations"]:
    logger.debug(f"edge \"{e['source']}\" --{e['relationship']}--> \"{e['target']}\"")

"""
### Graph Explorer Visualization Example

_Note that the visualization given below represents only a single example of the possible results generated by the LLM._

Visualization for the relationship:
```
"alice" --recommends--> "sci-fi"
"alice" --dislikes--> "thriller_movies"
"alice" --loves--> "sci-fi_movies"
"alice" --plans_to_watch--> "movie"
"alice" --avoids--> "thriller"
"thriller" --type_of--> "movie"
"movie" --can_be--> "engaging"
"sci-fi" --type_of--> "movie"
```

![neptune-example-visualization-4.png](./neptune-example-visualization-4.png)

## Search memories

Search all memories for "what does alice love?".  Since "alice" the user, this will search for a relationship that fits the users love of "sci-fi" movies and dislike of "thriller" movies.
"""
logger.info("### Graph Explorer Visualization Example")

search_results = m.search("what does alice love?", user_id=user_id)
for result in search_results["results"]:
    logger.debug(f"\"{result['memory']}\" [score: {result['score']}]")
for relation in search_results["relations"]:
    logger.debug(f"{relation}")

m.delete_all(user_id)
m.reset()

"""
## Conclusion

In this example we demonstrated how an AWS tech stack can be used to store and retrieve memory context. Bedrock LLM models can be used to interpret given conversations.  OpenSearch can store text chunks with vector embeddings. Neptune Analytics can store the text chunks in a graph format with relationship entities.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)