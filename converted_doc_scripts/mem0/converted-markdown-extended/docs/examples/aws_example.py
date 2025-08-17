from jet.logger import CustomLogger
from mem0.memory.main import Memory
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
import boto3
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
title: Amazon Stack: AWS Bedrock, AOSS, and Neptune Analytics
---

This example demonstrates how to configure and use the `mem0ai` SDK with **AWS Bedrock**, **OpenSearch Service (AOSS)**, and **AWS Neptune Analytics** for persistent memory capabilities in Python.

## Installation

Install the required dependencies to include the Amazon data stack, including **boto3**, **opensearch-py**, and **langchain-aws**:
"""
logger.info("## Installation")

pip install "mem0ai[graph,extras]"

"""
## Environment Setup

Set your AWS environment variables:
"""
logger.info("## Environment Setup")


os.environ['AWS_REGION'] = 'us-west-2'
os.environ['AWS_ACCESS_KEY_ID'] = 'AK00000000000000000'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'AS00000000000000000'

logger.debug(os.environ['AWS_REGION'])
logger.debug(os.environ['AWS_ACCESS_KEY_ID'])
logger.debug(os.environ['AWS_SECRET_ACCESS_KEY'])

"""
## Configuration and Usage

This sets up Mem0 with:
- [AWS Bedrock for LLM](https://docs.mem0.ai/components/llms/models/aws_bedrock)
- [AWS Bedrock for embeddings](https://docs.mem0.ai/components/embedders/models/aws_bedrock#aws-bedrock)
- [OpenSearch as the vector store](https://docs.mem0.ai/components/vectordbs/dbs/opensearch)
- [Neptune Analytics as your graph store](https://docs.mem0.ai/open-source/graph_memory/overview#initialize-neptune-analytics).
"""
logger.info("## Configuration and Usage")


region = 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

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

m = Memory.from_config(config)

"""
## Usage

Reference [Notebook example](https://github.com/mem0ai/mem0/blob/main/examples/graph-db-demo/neptune-example.ipynb)

#### Add a memory:
"""
logger.info("## Usage")

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"})

"""
#### Search a memory:
"""
logger.info("#### Search a memory:")

relevant_memories = m.search(query, user_id="alice")

"""
#### Get all memories:
"""
logger.info("#### Get all memories:")

all_memories = m.get_all(user_id="alice")

"""
#### Get a specific memory:
"""
logger.info("#### Get a specific memory:")

memory = m.get(memory_id)

"""
---

## Conclusion

With Mem0 and AWS services like Bedrock, OpenSearch, and Neptune Analytics, you can build intelligent AI companions that remember, adapt, and personalize their responses over time. This makes them ideal for long-term assistants, tutors, or support bots with persistent memory and natural conversation abilities.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)