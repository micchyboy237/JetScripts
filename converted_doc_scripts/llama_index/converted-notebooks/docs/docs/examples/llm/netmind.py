from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.netmind import NetmindLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Netmind AI LLM

This notebook shows how to use `Netmind AI` as an LLM. Check out the full list of models [netmind.ai](https://www.netmind.ai/).

Visit https://www.netmind.ai/ and sign up to get an API key.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Netmind AI LLM")

# %pip install llama-index-llms-netmind

# !pip install llama-index


llm = NetmindLLM(
    model="meta-llama/Llama-3.3-70B-Instruct", api_key="your api key"
)

resp = llm.complete("Is 9.9 or 9.11 bigger?")

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