from jet.adapters.langchain.chat_ollama.chat_models import ChatOllama
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar_position: 1.5
---

# How to stream chat model responses


All [chat models](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) implement the [Runnable interface](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable), which comes with a **default** implementations of standard runnable methods (i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`, `astream_events`).

The **default** streaming implementation provides an`Iterator` (or `AsyncIterator` for asynchronous streaming) that yields a single value: the final output from the underlying chat model provider.

:::tip

The **default** implementation does **not** provide support for token-by-token streaming, but it ensures that the model can be swapped in for any other model as it supports the same standard interface.

:::

The ability to stream the output token-by-token depends on whether the provider has implemented proper streaming support.

See which [integrations support token-by-token streaming here](/docs/integrations/chat/).

## Sync streaming

Below we use a `|` to help visualize the delimiter between tokens.
"""
logger.info("# How to stream chat model responses")


chat = ChatOllama(model="llama3.2")
for chunk in chat.stream("Write me a 1 verse song about goldfish on the moon"):
    logger.debug(chunk.content, end="|", flush=True)

"""
## Async Streaming
"""
logger.info("## Async Streaming")


chat = ChatOllama(model="llama3.2")
for chunk in chat.stream("Write me a 1 verse song about goldfish on the moon"):
    logger.debug(chunk.content, end="|", flush=True)

"""
## Astream events

Chat models also support the standard [astream events](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) method.

This method is useful if you're streaming output from a larger LLM application that contains multiple steps (e.g., an LLM chain composed of a prompt, llm and parser).
"""
logger.info("## Astream events")


chat = ChatOllama(model="llama3.2")
idx = 0

for event in chat.stream_events(
    "Write me a 1 verse song about goldfish on the moon"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        logger.debug("...Truncated")
        break
    logger.debug(event)

logger.info("\n\n[DONE]", bright=True)