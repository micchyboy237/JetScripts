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
title: Configurations
icon: "gear"
iconType: "solid"
---


Config in mem0 is a dictionary that specifies the settings for your embedding models. It allows you to customize the behavior and connection details of your chosen embedder.

## How to define configurations?

The config is defined as an object (or dictionary) with two main keys:
- `embedder`: Specifies the embedder provider and its configuration
  - `provider`: The name of the embedder (e.g., "openai", "ollama")
  - `config`: A nested object or dictionary containing provider-specific settings


## How to use configurations?

Here's a general example of how to use the config with mem0:

<CodeGroup>
"""
logger.info("## How to define configurations?")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "embedder": {
        "provider": "your_chosen_provider",
        "config": {
        }
    }
}

m = Memory.from_config(config)
m.add("Your text here", user_id="user", metadata={"category": "example"})

"""

"""


config = {
  embedder: {
    provider: 'openai',
    config: {
#       apiKey: process.env.OPENAI_API_KEY || '',
      model: 'mxbai-embed-large',
    },
  },
}

memory = new Memory(config)
async def run_async_code_444a0475():
    await memory.add("Your text here", { userId: "user", metadata: { category: "example" } })
    return 
 = asyncio.run(run_async_code_444a0475())
logger.success(format_json())

"""
</CodeGroup>

## Why is Config Needed?

Config is essential for:
1. Specifying which embedding model to use.
2. Providing necessary connection details (e.g., model, api_key, embedding_dims).
3. Ensuring proper initialization and connection to your chosen embedder.

## Master List of All Params in Config

Here's a comprehensive list of all parameters that can be used across different embedders:

<Tabs>
<Tab title="Python">
| Parameter | Description | Provider |
|-----------|-------------|----------|
| `model` | Embedding model to use | All |
| `api_key` | API key of the provider | All |
| `embedding_dims` | Dimensions of the embedding model | All |
| `http_client_proxies` | Allow proxy server settings | All |
| `ollama_base_url` | Base URL for the Ollama embedding model | Ollama |
| `model_kwargs` | Key-Value arguments for the Huggingface embedding model | Huggingface |
| `azure_kwargs` | Key-Value arguments for the AzureMLX embedding model | Azure MLX |
| `openai_base_url`    | Base URL for MLX API                       | MLX            |
| `vertex_credentials_json` | Path to the Google Cloud credentials JSON file for VertexAI                       | VertexAI            |
| `memory_add_embedding_type` | The type of embedding to use for the add memory action                       | VertexAI            |
| `memory_update_embedding_type` | The type of embedding to use for the update memory action                       | VertexAI            |
| `memory_search_embedding_type` | The type of embedding to use for the search memory action                       | VertexAI            |
| `lmstudio_base_url` | Base URL for LM Studio API                    | LM Studio         |
</Tab>
<Tab title="TypeScript">
| Parameter | Description | Provider |
|-----------|-------------|----------|
| `model` | Embedding model to use | All |
| `apiKey` | API key of the provider | All |
| `embeddingDims` | Dimensions of the embedding model | All |
</Tab>
</Tabs>

## Supported Embedding Models

For detailed information on configuring specific embedders, please visit the [Embedding Models](./models) section. There you'll find information for each supported embedder with provider-specific usage examples and configuration details.
"""
logger.info("## Why is Config Needed?")

logger.info("\n\n[DONE]", bright=True)