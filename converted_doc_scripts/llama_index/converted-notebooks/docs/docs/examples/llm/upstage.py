import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.upstage import Upstage
from pydantic import BaseModel
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/upstage.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Upstage

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Upstage")

# %pip install llama-index-llms-upstage llama-index

"""
## Basic Usage

#### Call `complete` with a prompt
"""
logger.info("## Basic Usage")


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


llm = Upstage(
    model="solar-mini",
)

resp = llm.complete("Paul Graham is ")

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
## Function Calling

Upstage models have native support for function calling. This conveniently integrates with LlamaIndex tool abstractions, letting you plug in any arbitrary Python function to the LLM.
"""
logger.info("## Function Calling")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


tool = FunctionTool.from_defaults(fn=generate_song)


llm = Upstage()
response = llm.predict_and_call([tool], "Generate a song")
logger.debug(str(response))

"""
We can also do multiple function calling.
"""
logger.info("We can also do multiple function calling.")

llm = Upstage()
response = llm.predict_and_call(
    [tool],
    "Generate five songs from the Beatles",
    allow_parallel_tool_calls=True,
)
for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
## Async
"""
logger.info("## Async")


llm = Upstage()

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

"""
Async function calling is also supported.
"""
logger.info("Async function calling is also supported.")

llm = Upstage()
async def run_async_code_d4d0bf51():
    async def run_async_code_bf5a731b():
        response = llm.predict_and_call([tool], "Generate a song")
        return response
    response = asyncio.run(run_async_code_bf5a731b())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_d4d0bf51())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)