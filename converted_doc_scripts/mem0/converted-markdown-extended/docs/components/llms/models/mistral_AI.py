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
title: Mistral AI
---

To use mistral's models, please obtain the Mistral AI api key from their [console](https://console.mistral.ai/). Set the `MISTRAL_API_KEY` environment variable to use the model as given below in the example.

## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key" # used for embedding model
os.environ["MISTRAL_API_KEY"] = "your-api-key"

config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "open-mixtral-8x7b",
            "temperature": 0.1,
            "max_tokens": 2000,
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
    provider: 'mistral',
    config: {
      apiKey: process.env.MISTRAL_API_KEY || '',
      model: 'mistral-tiny-latest', // Or 'mistral-small-latest', 'mistral-medium-latest', etc.
      temperature: 0.1,
      maxTokens: 2000,
    },
  },
}

memory = new Memory(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
async def run_async_code_3ea57109():
    await memory.add(messages, { userId: "alice", metadata: { category: "movies" } })
    return 
 = asyncio.run(run_async_code_3ea57109())
logger.success(format_json())

"""
</CodeGroup>

## Config

All available parameters for the `litellm` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)