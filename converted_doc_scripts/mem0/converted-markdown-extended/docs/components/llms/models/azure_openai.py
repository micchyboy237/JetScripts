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

<Note> Mem0 Now Supports Azure MLX Models in TypeScript SDK </Note>

# To use Azure MLX models, you have to set the `LLM_AZURE_OPENAI_API_KEY`, `LLM_AZURE_ENDPOINT`, `LLM_AZURE_DEPLOYMENT` and `LLM_AZURE_API_VERSION` environment variables. You can obtain the Azure API key from the [Azure](https://azure.microsoft.com/).

> **Note**: The following are currently unsupported with reasoning models `Parallel tool calling`,`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens`


## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key" # used for embedding model

# os.environ["LLM_AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["LLM_AZURE_DEPLOYMENT"] = "your-deployment-name"
os.environ["LLM_AZURE_ENDPOINT"] = "your-api-base-url"
os.environ["LLM_AZURE_API_VERSION"] = "version-to-use"

config = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": "your-deployment-name",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": "",
                  "api_version": "",
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
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""

"""


config = {
  llm: {
    provider: 'azure_openai',
    config: {
#       apiKey: process.env.AZURE_OPENAI_API_KEY || '',
      modelProperties: {
        endpoint: 'https://your-api-base-url',
        deployment: 'your-deployment-name',
        modelName: 'your-model-name',
        apiVersion: 'version-to-use',
      },
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


We also support the new [MLX structured-outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) model. Typescript SDK does not support the `azure_openai_structured` model yet.
"""
logger.info("We also support the new [MLX structured-outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) model. Typescript SDK does not support the `azure_openai_structured` model yet.")


# os.environ["LLM_AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["LLM_AZURE_DEPLOYMENT"] = "your-deployment-name"
os.environ["LLM_AZURE_ENDPOINT"] = "your-api-base-url"
os.environ["LLM_AZURE_API_VERSION"] = "version-to-use"

config = {
    "llm": {
        "provider": "azure_openai_structured",
        "config": {
            "model": "your-deployment-name",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": "",
                  "api_version": "",
                  "azure_endpoint": "",
                  "api_key": "",
                  "default_headers": {
                    "CustomHeader": "your-custom-header",
                  }
              }
        }
    }
}

"""
## Config

All available parameters for the `azure_openai` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)