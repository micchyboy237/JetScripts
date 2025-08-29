from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.cohere import Cohere
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/cohere.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cohere

## Basic Usage

#### Call `complete` with a prompt

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Cohere")

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-cohere

# !pip install llama-index


api_key = "Your api key"
resp = Cohere(api_key=api_key).complete("Paul Graham is ")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = Cohere(api_key=api_key).chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = Cohere(api_key=api_key)
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


llm = Cohere(api_key=api_key)
messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = Cohere(model="command", api_key=api_key)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
## Async
"""
logger.info("## Async")


llm = Cohere(model="command", api_key=api_key)

resp = llm.complete("Paul Graham is ")
logger.success(format_json(resp))
logger.success(format_json(resp))

logger.debug(resp)

resp = llm.stream_complete("Paul Graham is ")
logger.success(format_json(resp))
logger.success(format_json(resp))

async for delta in resp:
    logger.debug(delta.delta, end="")

"""
## Set API Key at a per-instance level
If desired, you can have separate LLM instances use separate API keys.
"""
logger.info("## Set API Key at a per-instance level")


llm_good = Cohere(api_key=api_key)
llm_bad = Cohere(model="command", api_key="BAD_KEY")

resp = llm_good.complete("Paul Graham is ")
logger.debug(resp)

resp = llm_bad.complete("Paul Graham is ")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)