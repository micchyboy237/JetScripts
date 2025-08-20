import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.paieas import PaiEas
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


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# AlibabaCloud-PaiEas

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# AlibabaCloud-PaiEas")

# %pip install llama-index-llms-paieas

# !pip install llama-index

"""
## Basic Usage

You will need to get an API key and url from [AlibabaCloud PAI Eas](https://help.aliyun.com/zh/pai/use-cases/deploy-llm-in-eas?spm=5176.pai-console-inland.help.dexternal.107e642dLd2e9J). Once you have one, you can either pass it explicity to the model, or use the `PAIEAS_API_KEY` and `PAIEAS_API_BASE` environment variable.
"""
logger.info("## Basic Usage")

# !export PAIEAS_API_KEY=your_service_token
# !export PAIEAS_API_BASE=your_access_address

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")


llm = PaiEas()
resp = llm.complete("Write a poem about a magic backpack")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Async
"""
logger.info("## Async")

async def run_async_code_c3ecd675():
    async def run_async_code_a989c387():
        resp = llm.complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_a989c387())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_c3ecd675())
logger.success(format_json(resp))

logger.debug(resp)

async def run_async_code_240f4fad():
    async def run_async_code_506ce1e2():
        resp = llm.stream_complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_506ce1e2())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_240f4fad())
logger.success(format_json(resp))

async for delta in resp:
    logger.debug(delta.delta, end="")

logger.info("\n\n[DONE]", bright=True)