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
[Pinecone](https://www.pinecone.io/) is a fully managed vector database designed for machine learning applications, offering high performance vector search with low latency at scale. It's particularly well-suited for semantic search, recommendation systems, and other AI-powered applications.

> **New**: Pinecone integration now supports custom namespaces! Use the `namespace` parameter to logically separate data within the same index. This is especially useful for multi-tenant or multi-user applications.

> **Note**: Before configuring Pinecone, you need to select an embedding model (e.g., MLX, Cohere, or custom models) and ensure the `embedding_model_dims` in your config matches your chosen model's dimensions. For example, MLX's mxbai-embed-large uses 1536 dimensions.

### Usage
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"
os.environ["PINECONE_API_KEY"] = "your-api-key"

config = {
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "testing",
            "embedding_model_dims": 1536,  # Matches MLX's mxbai-embed-large
            "namespace": "my-namespace", # Optional: specify a namespace for multi-tenancy
            "serverless_config": {
                "cloud": "aws",  # Choose between 'aws' or 'gcp' or 'azure'
                "region": "us-east-1"
            },
            "metric": "cosine"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Config

Here are the parameters available for configuring Pinecone:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `collection_name` | Name of the index/collection | Required |
| `embedding_model_dims` | Dimensions of the embedding model (must match your chosen embedding model) | Required |
| `client` | Existing Pinecone client instance | `None` |
| `api_key` | API key for Pinecone | Environment variable: `PINECONE_API_KEY` |
| `environment` | Pinecone environment | `None` |
| `serverless_config` | Configuration for serverless deployment (AWS or GCP or Azure) | `None` |
| `pod_config` | Configuration for pod-based deployment | `None` |
| `hybrid_search` | Whether to enable hybrid search | `False` |
| `metric` | Distance metric for vector similarity | `"cosine"` |
| `batch_size` | Batch size for operations | `100` |
| `namespace` | Namespace for the collection, useful for multi-tenancy. | `None` |

> **Important**: You must choose either `serverless_config` or `pod_config` for your deployment, but not both.

#### Serverless Config Example
"""
logger.info("### Config")

config = {
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "memory_index",
            "embedding_model_dims": 1536,  # For MLX's mxbai-embed-large
            "namespace": "my-namespace",  # Optional: custom namespace
            "serverless_config": {
                "cloud": "aws",  # or "gcp" or "azure"
                "region": "us-east-1"  # Choose appropriate region
            }
        }
    }
}

"""
#### Pod Config Example
"""
logger.info("#### Pod Config Example")

config = {
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "memory_index",
            "embedding_model_dims": 1536,  # For MLX's text-embedding-ada-002
            "namespace": "my-namespace",  # Optional: custom namespace
            "pod_config": {
                "environment": "gcp-starter",
                "replicas": 1,
                "pod_type": "starter"
            }
        }
    }
}

logger.info("\n\n[DONE]", bright=True)