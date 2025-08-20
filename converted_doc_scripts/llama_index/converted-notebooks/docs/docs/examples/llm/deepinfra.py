import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepinfra import DeepInfraLLM
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/deepinfra.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

"""
# DeepInfra

## Installation

First, install the necessary package:

```bash
%pip install llama-index-llms-deepinfra
```
"""
logger.info("# DeepInfra")

# %pip install llama-index-llms-deepinfra

"""
## Initialization

Set up the `DeepInfraLLM` class with your API key and desired parameters:
"""
logger.info("## Initialization")


llm = DeepInfraLLM(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",  # Default model name
    api_key="your-deepinfra-api-key",  # Replace with your DeepInfra API key
    temperature=0.5,
    max_tokens=50,
    additional_kwargs={"top_p": 0.9},
)

"""
## Synchronous Complete

Generate a text completion synchronously using the `complete` method:
"""
logger.info("## Synchronous Complete")

response = llm.complete("Hello World!")
logger.debug(response.text)

"""
## Synchronous Stream Complete

Generate a streaming text completion synchronously using the `stream_complete` method:
"""
logger.info("## Synchronous Stream Complete")

content = ""
for completion in llm.stream_complete("Once upon a time"):
    content += completion.delta
    logger.debug(completion.delta, end="")

"""
## Synchronous Chat

Generate a chat response synchronously using the `chat` method:
"""
logger.info("## Synchronous Chat")


messages = [
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)
logger.debug(chat_response.message.content)

"""
## Synchronous Stream Chat

Generate a streaming chat response synchronously using the `stream_chat` method:
"""
logger.info("## Synchronous Stream Chat")

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a story."),
]
content = ""
for chat_response in llm.stream_chat(messages):
    content += chat_response.message.delta
    logger.debug(chat_response.message.delta, end="")

"""
## Asynchronous Complete

Generate a text completion asynchronously using the `acomplete` method:
"""
logger.info("## Asynchronous Complete")

async def async_complete():
    async def run_async_code_e062ceb6():
        async def run_async_code_8b4010de():
            response = llm.complete("Hello Async World!")
            return response
        response = asyncio.run(run_async_code_8b4010de())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_e062ceb6())
    logger.success(format_json(response))
    logger.debug(response.text)


asyncio.run(async_complete())

"""
## Asynchronous Stream Complete

Generate a streaming text completion asynchronously using the `astream_complete` method:
"""
logger.info("## Asynchronous Stream Complete")

async def async_stream_complete():
    content = ""
    async def run_async_code_1d49cbae():
        async def run_async_code_14611d6a():
            response = llm.stream_complete("Once upon an time")
            return response
        response = asyncio.run(run_async_code_14611d6a())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_1d49cbae())
    logger.success(format_json(response))
    async for completion in response:
        content += completion.delta
        logger.debug(completion.delta, end="")


asyncio.run(async_stream_complete())

"""
## Asynchronous Chat

Generate a chat response asynchronously using the `achat` method:
"""
logger.info("## Asynchronous Chat")

async def async_chat():
    messages = [
        ChatMessage(role="user", content="Tell me an async joke."),
    ]
    async def run_async_code_b5901de8():
        async def run_async_code_5d238f30():
            chat_response = llm.chat(messages)
            return chat_response
        chat_response = asyncio.run(run_async_code_5d238f30())
        logger.success(format_json(chat_response))
        return chat_response
    chat_response = asyncio.run(run_async_code_b5901de8())
    logger.success(format_json(chat_response))
    logger.debug(chat_response.message.content)


asyncio.run(async_chat())

"""
## Asynchronous Stream Chat

Generate a streaming chat response asynchronously using the `astream_chat` method:
"""
logger.info("## Asynchronous Stream Chat")

async def async_stream_chat():
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Tell me an async story."),
    ]
    content = ""
    async def run_async_code_b338540c():
        async def run_async_code_b0b33fe3():
            response = llm.stream_chat(messages)
            return response
        response = asyncio.run(run_async_code_b0b33fe3())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_b338540c())
    logger.success(format_json(response))
    async for chat_response in response:
        content += chat_response.message.delta
        logger.debug(chat_response.message.delta, end="")


asyncio.run(async_stream_chat())

"""
---

For any questions or feedback, please contact us at [feedback@deepinfra.com](mailto:feedback@deepinfra.com).
"""
logger.info("For any questions or feedback, please contact us at [feedback@deepinfra.com](mailto:feedback@deepinfra.com).")

logger.info("\n\n[DONE]", bright=True)