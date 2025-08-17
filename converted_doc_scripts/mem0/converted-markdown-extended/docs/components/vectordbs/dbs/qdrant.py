import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from 'mem0ai/oss'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[Qdrant](https://qdrant.tech/) is an open-source vector search engine. It is designed to work with large-scale datasets and provides a high-performance search engine for vector data.

### Usage

<CodeGroup>
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
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

"""


config = {
  vectorStore: {
    provider: 'qdrant',
    config: {
      collectionName: 'memories',
      embeddingModelDims: 1536,
      host: 'localhost',
      port: 6333,
    },
  },
}

memory = new Memory(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
async def run_async_code_3ea57109():
    await memory.add(messages, { userId: "alice", metadata: { category: "movies" } })
    return 
 = asyncio.run(run_async_code_3ea57109())
logger.success(format_json())

"""
</CodeGroup>

### Config

Let's see the available parameters for the `qdrant` config:

<Tabs>
<Tab title="Python">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `collection_name` | The name of the collection to store the vectors | `mem0` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `client` | Custom client for qdrant | `None` |
| `host` | The host where the qdrant server is running | `None` |
| `port` | The port where the qdrant server is running | `None` |
| `path` | Path for the qdrant database | `/tmp/qdrant` |
| `url` | Full URL for the qdrant server | `None` |
| `api_key` | API key for the qdrant server | `None` |
| `on_disk` | For enabling persistent storage | `False` |
</Tab>
<Tab title="TypeScript">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `collectionName` | The name of the collection to store the vectors | `mem0` |
| `embeddingModelDims` | Dimensions of the embedding model | `1536` |
| `host` | The host where the Qdrant server is running | `None` |
| `port` | The port where the Qdrant server is running | `None` |
| `path` | Path for the Qdrant database | `/tmp/qdrant` |
| `url` | Full URL for the Qdrant server | `None` |
| `apiKey` | API key for the Qdrant server | `None` |
| `onDisk` | For enabling persistent storage | `False` |
</Tab>
</Tabs>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)