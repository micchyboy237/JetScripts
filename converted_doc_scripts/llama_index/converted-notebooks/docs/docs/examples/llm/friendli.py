from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.friendli import Friendli
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

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

resp = llm.chat([message])
logger.success(format_json(resp))
logger.success(format_json(resp))

logger.debug(resp)

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")

resp = llm.stream_chat([message])
logger.success(format_json(resp))
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

resp = llm.complete(prompt)
logger.success(format_json(resp))
logger.success(format_json(resp))

logger.debug(resp)

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")

resp = llm.stream_complete(prompt)
logger.success(format_json(resp))
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