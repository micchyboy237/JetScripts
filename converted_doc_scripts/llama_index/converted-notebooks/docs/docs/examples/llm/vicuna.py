from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.replicate import Replicate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/vicuna.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Replicate - Vicuna 13B

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Replicate - Vicuna 13B")

# %pip install llama-index-llms-replicate

# !pip install llama-index

"""
Make sure you have the `REPLICATE_API_TOKEN` environment variable set.  
If you don't have one yet, go to https://replicate.com/ to obtain one.
"""
logger.info("Make sure you have the `REPLICATE_API_TOKEN` environment variable set.")


os.environ["REPLICATE_API_TOKEN"] = "<your API key>"

"""
## Basic Usage

We showcase the "vicuna-13b" model, which you can play with directly at: https://replicate.com/replicate/vicuna-13b
"""
logger.info("## Basic Usage")


llm = Replicate(
    model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")

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

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = Replicate(
    model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    temperature=0.9,
    max_tokens=32,
)

resp = llm.complete("Who is Paul Graham?")

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)