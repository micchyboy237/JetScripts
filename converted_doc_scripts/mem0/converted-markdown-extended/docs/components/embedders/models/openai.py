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
---
title: MLX
---

# To use MLX embedding models, set the `OPENAI_API_KEY` environment variable. You can obtain the MLX API key from the [MLX Platform](https://platform.openai.com/account/api-keys).

### Usage

<CodeGroup>
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "your_api_key"

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large"
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
m.add(messages, user_id="john")

"""

"""


config = {
  embedder: {
    provider: 'openai',
    config: {
      apiKey: 'your-openai-api-key',
      model: 'text-embedding-3-large',
    },
  },
}

memory = new Memory(config)
async def run_async_code_80bb59fc():
    await memory.add("I'm visiting Paris", { userId: "john" })
    return 
 = asyncio.run(run_async_code_80bb59fc())
logger.success(format_json())

"""
</CodeGroup>

### Config

Here are the parameters available for configuring MLX embedder:

<Tabs>
<Tab title="Python">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the embedding model to use | `mxbai-embed-large` |
| `embedding_dims` | Dimensions of the embedding model | `1536` |
| `api_key` | The MLX API key | `None` |
</Tab>
<Tab title="TypeScript">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the embedding model to use | `mxbai-embed-large` |
| `embeddingDims` | Dimensions of the embedding model | `1536` |
| `apiKey` | The MLX API key | `None` |
</Tab>
</Tabs>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)