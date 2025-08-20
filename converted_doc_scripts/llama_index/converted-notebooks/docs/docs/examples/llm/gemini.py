import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Gemini

**NOTE:** Gemini has largely been replaced by Google GenAI. Visit the [Google GenAI page](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/) for the latest examples and documentation.

In this notebook, we show how to use the Gemini text models from Google in LlamaIndex. Check out the [Gemini site](https://ai.google.dev/) or the [announcement](https://deepmind.google/technologies/gemini/).

If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the Gemini Python SDK.
"""
logger.info("# Gemini")

# %pip install llama-index-llms-gemini llama-index

"""
## Basic Usage

You will need to get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey). Once you have one, you can either pass it explicity to the model, or use the `GOOGLE_API_KEY` environment variable.
"""
logger.info("## Basic Usage")

# %env GOOGLE_API_KEY=...


GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


llm = Gemini(
    model="models/gemini-1.5-flash",
)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")


resp = llm.complete("Write a poem about a magic backpack")
logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.chat(messages)
logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

resp = llm.stream_complete(
    "The story of Sourcrust, the bread creature, is really interesting. It all started when..."
)

for r in resp:
    logger.debug(r.text, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Using other models

The [Gemini model site](https://ai.google.dev/models) lists the models that are currently available, along with their capabilities. You can also use the API to find suitable models.
"""
logger.info("## Using other models")


for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        logger.debug(m.name)


llm = Gemini(model="models/gemini-pro")

resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
logger.debug(resp)

"""
## Asynchronous API
"""
logger.info("## Asynchronous API")


llm = Gemini()

async def run_async_code_e902d621():
    async def run_async_code_e8ebda3c():
        resp = llm.complete("Llamas are famous for ")
        return resp
    resp = asyncio.run(run_async_code_e8ebda3c())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_e902d621())
logger.success(format_json(resp))
logger.debug(resp)

async def run_async_code_1624af0b():
    async def run_async_code_f3edd59f():
        resp = llm.stream_complete("Llamas are famous for ")
        return resp
    resp = asyncio.run(run_async_code_f3edd59f())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_1624af0b())
logger.success(format_json(resp))
async for chunk in resp:
    logger.debug(chunk.text, end="")

logger.info("\n\n[DONE]", bright=True)