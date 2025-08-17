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
title: Azure MLX
---

# To use Azure MLX embedding models, set the `EMBEDDING_AZURE_OPENAI_API_KEY`, `EMBEDDING_AZURE_DEPLOYMENT`, `EMBEDDING_AZURE_ENDPOINT` and `EMBEDDING_AZURE_API_VERSION` environment variables. You can obtain the Azure MLX API key from the Azure.

### Usage

<CodeGroup>
"""
logger.info("### Usage")


# os.environ["EMBEDDING_AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["EMBEDDING_AZURE_DEPLOYMENT"] = "your-deployment-name"
os.environ["EMBEDDING_AZURE_ENDPOINT"] = "your-api-base-url"
os.environ["EMBEDDING_AZURE_API_VERSION"] = "version-to-use"

# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM


config = {
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": "text-embedding-3-large"
            "azure_kwargs": {
                  "api_version": "",
                  "azure_deployment": "",
                  "azure_endpoint": "",
                  "api_key": "",
                  "default_headers": {
                    "CustomHeader": "your-custom-header",
                  }
              }
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
        provider: "azure_openai",
        config: {
            model: "text-embedding-3-large",
            modelProperties: {
                endpoint: "your-api-base-url",
                deployment: "your-deployment-name",
                apiVersion: "version-to-use",
            }
        }
    }
}

memory = new Memory(config)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

async def run_async_code_bcb4285b():
    await memory.add(messages, { userId: "john" })
    return 
 = asyncio.run(run_async_code_bcb4285b())
logger.success(format_json())

"""
</CodeGroup>

### Config

Here are the parameters available for configuring Azure MLX embedder:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the embedding model to use | `mxbai-embed-large` |
| `embedding_dims` | Dimensions of the embedding model | `1536` |
| `azure_kwargs` | The Azure MLX configs | `config_keys` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)