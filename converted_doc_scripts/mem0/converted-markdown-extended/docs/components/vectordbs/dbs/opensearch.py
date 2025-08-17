from jet.logger import CustomLogger
from mem0 import Memory
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
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
[OpenSearch](https://opensearch.org/) is an enterprise-grade search and observability suite that brings order to unstructured data at scale. OpenSearch supports k-NN (k-Nearest Neighbors) and allows you to store and retrieve high-dimensional vector embeddings efficiently.

### Installation

OpenSearch support requires additional dependencies. Install them with:
"""
logger.info("### Installation")

pip install opensearch-py

"""
### Prerequisites

Before using OpenSearch with Mem0, you need to set up a collection in AWS OpenSearch Service.

#### AWS OpenSearch Service
You can create a collection through the AWS Console:
- Navigate to [OpenSearch Service Console](https://console.aws.amazon.com/aos/home)
- Click "Create collection"
- Select "Serverless collection" and then enable "Vector search" capabilities
- Once created, note the endpoint URL (host) for your configuration


### Usage
"""
logger.info("### Prerequisites")


region = 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

config = {
    "vector_store": {
        "provider": "opensearch",
        "config": {
            "collection_name": "mem0",
            "host": "your-domain.us-west-2.aoss.amazonaws.com",
            "port": 443,
            "http_auth": auth,
            "embedding_model_dims": 1024,
            "connection_class": RequestsHttpConnection,
            "pool_maxsize": 20,
            "use_ssl": True,
            "verify_certs": True
        }
    }
}

"""
### Add Memories
"""
logger.info("### Add Memories")

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Search Memories
"""
logger.info("### Search Memories")

results = m.search("What kind of movies does Alice like?", user_id="alice")

"""
### Features

- Fast and Efficient Vector Search
- Can be deployed on-premises, in containers, or on cloud platforms like AWS OpenSearch Service.
- Multiple Authentication and Security Methods (Basic Authentication, API Keys, LDAP, SAML, and OpenID Connect)
- Automatic index creation with optimized mappings for vector search
- Memory Optimization through Disk-Based Vector Search and Quantization
- Real-Time Analytics and Observability
"""
logger.info("### Features")

logger.info("\n\n[DONE]", bright=True)