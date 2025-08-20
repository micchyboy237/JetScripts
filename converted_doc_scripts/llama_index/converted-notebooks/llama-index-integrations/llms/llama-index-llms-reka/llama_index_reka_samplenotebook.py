import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.reka import RekaLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

pip install llama-index-llms-reka

"""
To obtain API key, please visit [https://platform.reka.ai/](https://platform.reka.ai/)

# Chat completion
"""
logger.info("# Chat completion")


api_key = os.getenv("REKA_API_KEY")
reka_llm = RekaLLM(
    model="reka-flash",
    api_key=api_key,
)

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
]
response = reka_llm.chat(messages)
logger.debug(response.message.content)

prompt = "The capital of France is"
response = reka_llm.complete(prompt)
logger.debug(response.text)

"""
# Streaming example
"""
logger.info("# Streaming example")

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(
        role=MessageRole.USER, content="List the first 5 planets in the solar system."
    ),
]
for chunk in reka_llm.stream_chat(messages):
    logger.debug(chunk.delta, end="", flush=True)

prompt = "List the first 5 planets in the solar system:"
for chunk in reka_llm.stream_complete(prompt):
    logger.debug(chunk.delta, end="", flush=True)

"""
# Async use cases (chat/completion)
"""
logger.info("# Async use cases (chat/completion)")

async def main():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    async def run_async_code_beb11b61():
        async def run_async_code_e096bdef():
            response = reka_llm.chat(messages)
            return response
        response = asyncio.run(run_async_code_e096bdef())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_beb11b61())
    logger.success(format_json(response))
    logger.debug(response.message.content)

    prompt = "The largest planet in our solar system is"
    async def run_async_code_8107d7a4():
        async def run_async_code_51999721():
            response = reka_llm.complete(prompt)
            return response
        response = asyncio.run(run_async_code_51999721())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_8107d7a4())
    logger.success(format_json(response))
    logger.debug(response.text)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    async def run_async_code_03c5880e():
        for chunk in reka_llm.stream_chat(messages):
        return 
     = asyncio.run(run_async_code_03c5880e())
    logger.success(format_json())
        logger.debug(chunk.delta, end="", flush=True)

    prompt = "List the first 5 elements in the periodic table:"
    async def run_async_code_3030d3f1():
        for chunk in reka_llm.stream_complete(prompt):
        return 
     = asyncio.run(run_async_code_3030d3f1())
    logger.success(format_json())
        logger.debug(chunk.delta, end="", flush=True)


async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)