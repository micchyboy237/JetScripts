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
---
title: DeepSeek
---

To use DeepSeek LLM models, you have to set the `DEEPSEEK_API_KEY` environment variable. You can also optionally set `DEEPSEEK_API_BASE` if you need to use a different API endpoint (defaults to "https://api.deepseek.com").

## Usage
"""
logger.info("## Usage")


os.environ["DEEPSEEK_API_KEY"] = "your-api-key"
# os.environ["OPENAI_API_KEY"] = "your-api-key" # for embedder model

config = {
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": "deepseek-chat",  # default model
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
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
You can also configure the API base URL in the config:
"""
logger.info("You can also configure the API base URL in the config:")

config = {
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": "deepseek-chat",
            "deepseek_base_url": "https://your-custom-endpoint.com",
            "api_key": "your-api-key"  # alternatively to using environment variable
        }
    }
}

"""
## Config

All available parameters for the `deepseek` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)