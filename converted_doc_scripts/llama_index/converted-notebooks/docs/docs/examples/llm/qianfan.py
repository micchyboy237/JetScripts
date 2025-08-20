import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.qianfan import Qianfan
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Client of Baidu Intelligent Cloud's Qianfan LLM Platform

Baidu Intelligent Cloud's Qianfan LLM Platform offers API services for all Baidu LLMs, such as ERNIE-3.5-8K and ERNIE-4.0-8K. It also provides a small number of open-source LLMs like Llama-2-70b-chat.

Before using the chat client, you need to activate the LLM service on the Qianfan LLM Platform console's [online service](https://console.bce.baidu.com/qianfan/ais/console/onlineService) page. Then, Generate an Access Key and a Secret Key in the [Security Authentication](https://console.bce.baidu.com/iam/#/iam/accesslist) page of the console.

## Installation

Install the necessary package:
"""
logger.info("# Client of Baidu Intelligent Cloud's Qianfan LLM Platform")

# %pip install llama-index-llms-qianfan

"""
## Initialization
"""
logger.info("## Initialization")


access_key = "XXX"
secret_key = "XXX"
model_name = "ERNIE-Speed-8K"
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
context_window = 8192
llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

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
    content += chat_response.delta
    logger.debug(chat_response.delta, end="")

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
        content += chat_response.delta
        logger.debug(chat_response.delta, end="")


asyncio.run(async_stream_chat())

logger.info("\n\n[DONE]", bright=True)