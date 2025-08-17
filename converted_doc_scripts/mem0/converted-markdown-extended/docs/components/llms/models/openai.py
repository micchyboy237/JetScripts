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

# To use MLX LLM models, you have to set the `OPENAI_API_KEY` environment variable. You can obtain the MLX API key from the [MLX Platform](https://platform.openai.com/account/api-keys).

> **Note**: The following are currently unsupported with reasoning models `Parallel tool calling`,`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens`

## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
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
    provider: 'openai',
    config: {
#       apiKey: process.env.OPENAI_API_KEY || '',
      model: 'gpt-4-turbo-preview',
      temperature: 0.2,
      maxTokens: 1500,
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

We also support the new [MLX structured-outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) model.
"""
logger.info("We also support the new [MLX structured-outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) model.")


# os.environ["OPENAI_API_KEY"] = "your-api-key"

config = {
    "llm": {
        "provider": "openai_structured",
        "config": {
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.0,
        }
    }
}

m = Memory.from_config(config)

"""
## Config

All available parameters for the `openai` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)