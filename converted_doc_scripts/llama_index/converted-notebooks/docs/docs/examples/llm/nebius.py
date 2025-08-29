from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.nebius import NebiusLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/nebius.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Nebius LLMs

This notebook demonstrates how to use LLMs from [Nebius AI Studio](https://studio.nebius.ai/) with LlamaIndex. Nebius AI Studio implements all state-of-the-art LLMs available for commercial use.

First, let's install LlamaIndex and dependencies of Nebius AI Studio.
"""
logger.info("# Nebius LLMs")

# %pip install llama-index-llms-nebius llama-index

"""
Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys)."
"""
logger.info("Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys)."")


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")  # NEBIUS_API_KEY = ""


llm = NebiusLLM(
    api_key=NEBIUS_API_KEY, model="meta-llama/Llama-3.3-70B-Instruct-fast"
)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")

response = llm.complete("Amsterdam is the capital of ")
logger.debug(response)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant."),
    ChatMessage(
        role="user",
        content="Answer briefly: who is Wall-e?",
    ),
]
response = llm.chat(messages)
logger.debug(response)

"""
### Streaming

#### Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Amsterdam is the capital of ")
for r in response:
    logger.debug(r.delta, end="")

"""
#### Using `stream_chat` with a list of messages
"""
logger.info("#### Using `stream_chat` with a list of messages")


messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant."),
    ChatMessage(
        role="user",
        content="Answer briefly: who is Wall-e?",
    ),
]
response = llm.stream_chat(messages)
for r in response:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)