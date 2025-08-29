from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.together import TogetherLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/together.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Together AI LLM

This notebook shows how to use `Together AI` as an LLM. Together AI provides access to many state-of-the-art LLM models. Check out the full list of models [here](https://docs.together.ai/docs/inference-models).

Visit https://together.ai and sign up to get an API key.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Together AI LLM")

# %pip install llama-index-llms-together

# !pip install llama-index


llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="your_api_key"
)

resp = llm.complete("Who is Paul Graham?")

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
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Who is Paul Graham?")

for r in response:
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

logger.info("\n\n[DONE]", bright=True)