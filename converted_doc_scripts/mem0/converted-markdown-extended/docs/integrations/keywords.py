from jet.logger import CustomLogger
from mem0 import Memory
from openai import MLX
import json
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
title: Keywords AI
---

Build AI applications with persistent memory and comprehensive LLM observability by integrating Mem0 with Keywords AI.

## Overview

Mem0 is a self-improving memory layer for LLM applications, enabling personalized AI experiences that save costs and delight users. Keywords AI provides complete LLM observability.

Combining Mem0 with Keywords AI allows you to:
1. Add persistent memory to your AI applications
2. Track interactions across sessions
3. Monitor memory usage and retrieval with Keywords AI observability
4. Optimize token usage and reduce costs

<Note>
You can get your Mem0 API key, user_id, and org_id from the [Mem0 dashboard](https://app.mem0.ai/). These are required for proper integration.
</Note>

## Setup and Configuration

Install the necessary libraries:
"""
logger.info("## Overview")

pip install mem0 keywordsai-sdk

"""
Set up your environment variables:
"""
logger.info("Set up your environment variables:")


os.environ["MEM0_API_KEY"] = "your-mem0-api-key"
os.environ["KEYWORDSAI_API_KEY"] = "your-keywords-api-key"
os.environ["KEYWORDSAI_BASE_URL"] = "https://api.keywordsai.co/api/"

"""
## Basic Integration Example

Here's a simple example of using Mem0 with Keywords AI:
"""
logger.info("## Basic Integration Example")


api_key = os.getenv("MEM0_API_KEY")
keywordsai_api_key = os.getenv("KEYWORDSAI_API_KEY")
base_url = os.getenv("KEYWORDSAI_BASE_URL") # "https://api.keywordsai.co/api/"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "llama-3.2-3b-instruct",
            "temperature": 0.0,
            "api_key": keywordsai_api_key,
            "openai_base_url": base_url,
        },
    }
}

memory = Memory.from_config(config_dict=config)

result = memory.add(
    "I like to take long walks on weekends.",
    user_id="alice",
    metadata={"category": "hobbies"},
)

logger.debug(result)

"""
## Advanced Integration with MLX SDK

For more advanced use cases, you can integrate Keywords AI with Mem0 through the MLX SDK:
"""
logger.info("## Advanced Integration with MLX SDK")


client = MLX(
    api_key=os.environ.get("KEYWORDSAI_API_KEY"),
    base_url=os.environ.get("KEYWORDSAI_BASE_URL"),
)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=messages,
    extra_body={
        "mem0_params": {
            "user_id": "test_user",
            "org_id": "org_1",
            "api_key": os.environ.get("MEM0_API_KEY"),
            "add_memories": {
                "messages": messages,
            },
        }
    },
)

logger.debug(json.dumps(response.model_dump(), indent=4))

"""
For detailed information on this integration, refer to the official [Keywords AI Mem0 integration documentation](https://docs.keywordsai.co/integration/development-frameworks/mem0).

## Key Features

1. **Memory Integration**: Store and retrieve relevant information from past interactions
2. **LLM Observability**: Track memory usage and retrieval patterns with Keywords AI
3. **Session Persistence**: Maintain context across multiple user sessions
4. **Cost Optimization**: Reduce token usage through efficient memory retrieval

## Conclusion

Integrating Mem0 with Keywords AI provides a powerful combination for building AI applications with persistent memory and comprehensive observability. This integration enables more personalized user experiences while providing insights into your application's memory usage.

## Help

For more information, refer to:
- [Keywords AI Documentation](https://docs.keywordsai.co)
- [Mem0 Platform]((https://app.mem0.ai/))

<Snippet file="get-help.mdx" />
"""
logger.info("## Key Features")

logger.info("\n\n[DONE]", bright=True)