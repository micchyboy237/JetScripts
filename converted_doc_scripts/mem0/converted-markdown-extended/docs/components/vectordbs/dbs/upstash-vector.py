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
[Upstash Vector](https://upstash.com/docs/vector) is a serverless vector database with built-in embedding models.

### Usage with Upstash embeddings

You can enable the built-in embedding models by setting `enable_embeddings` to `True`. This allows you to use Upstash's embedding models for vectorization.
"""
logger.info("### Usage with Upstash embeddings")


os.environ["UPSTASH_VECTOR_REST_URL"] = "..."
os.environ["UPSTASH_VECTOR_REST_TOKEN"] = "..."

config = {
    "vector_store": {
        "provider": "upstash_vector",
        "enable_embeddings": True,
    }
}

m = Memory.from_config(config)
m.add("Likes to play cricket on weekends", user_id="alice", metadata={"category": "hobbies"})

"""
<Note>
    Setting `enable_embeddings` to `True` will bypass any external embedding provider you have configured.
</Note>

### Usage with external embedding providers
"""
logger.info("### Usage with external embedding providers")


# os.environ["OPENAI_API_KEY"] = "..."
os.environ["UPSTASH_VECTOR_REST_URL"] = "..."
os.environ["UPSTASH_VECTOR_REST_TOKEN"] = "..."

config = {
    "vector_store": {
        "provider": "upstash_vector",
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large"
        },
    }
}

m = Memory.from_config(config)
m.add("Likes to play cricket on weekends", user_id="alice", metadata={"category": "hobbies"})

"""
### Config

Here are the parameters available for configuring Upstash Vector:

| Parameter           | Description                        | Default Value |
| ------------------- | ---------------------------------- | ------------- |
| `url`               | URL for the Upstash Vector index   | `None`        |
| `token`             | Token for the Upstash Vector index | `None`        |
| `client`            | An `upstash_vector.Index` instance | `None`        |
| `collection_name`   | The default namespace used         | `""`          |
| `enable_embeddings` | Whether to use Upstash embeddings  | `False`       |

<Note>
  When `url` and `token` are not provided, the `UPSTASH_VECTOR_REST_URL` and
  `UPSTASH_VECTOR_REST_TOKEN` environment variables are used.
</Note>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)