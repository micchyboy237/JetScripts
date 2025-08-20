import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.friendli import Friendli
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/friendli.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Friendli

## Basic Usage

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Friendli")

# %pip install llama-index-llms-friendli

# !pip install llama-index

# %env FRIENDLI_TOKEN=...



llm = Friendli()

"""
### Call `chat` with a list of messages
"""
logger.info("### Call `chat` with a list of messages")


message = ChatMessage(role=MessageRole.USER, content="Tell me a joke.")
resp = llm.chat([message])

logger.debug(resp)

"""
#### Streaming
"""
logger.info("#### Streaming")

resp = llm.stream_chat([message])
for r in resp:
    logger.debug(r.delta, end="")

"""
#### Async
"""
logger.info("#### Async")

async def run_async_code_35524c17():
    async def run_async_code_4ef68c2f():
        resp = llm.chat([message])
        return resp
    resp = asyncio.run(run_async_code_4ef68c2f())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_35524c17())
logger.success(format_json(resp))

logger.debug(resp)

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")

async def run_async_code_cc7afffd():
    async def run_async_code_bed1a751():
        resp = llm.stream_chat([message])
        return resp
    resp = asyncio.run(run_async_code_bed1a751())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_cc7afffd())
logger.success(format_json(resp))
async for r in resp:
    logger.debug(r.delta, end="")

"""
### Call `complete` with a prompt
"""
logger.info("### Call `complete` with a prompt")

prompt = "Draft a cover letter for a role in software engineering."
resp = llm.complete(prompt)

logger.debug(resp)

"""
#### Streaming
"""
logger.info("#### Streaming")

resp = llm.stream_complete(prompt)
for r in resp:
    logger.debug(r.delta, end="")

"""
#### Async
"""
logger.info("#### Async")

async def run_async_code_c0189c24():
    async def run_async_code_a12fb04b():
        resp = llm.complete(prompt)
        return resp
    resp = asyncio.run(run_async_code_a12fb04b())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_c0189c24())
logger.success(format_json(resp))

logger.debug(resp)

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")

async def run_async_code_bd86587b():
    async def run_async_code_f1192b9d():
        resp = llm.stream_complete(prompt)
        return resp
    resp = asyncio.run(run_async_code_f1192b9d())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_bd86587b())
logger.success(format_json(resp))
async for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = Friendli(model="llama-2-70b-chat")

resp = llm.chat([message])

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)