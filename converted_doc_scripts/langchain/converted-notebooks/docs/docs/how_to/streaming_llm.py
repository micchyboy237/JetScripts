from jet.adapters.langchain.chat_ollama import ChatOllama
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
# How to stream responses from an LLM

All `LLM`s implement the [Runnable interface](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable), which comes with **default** implementations of standard runnable methods (i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`, `astream_events`).

The **default** streaming implementations provide an`Iterator` (or `AsyncIterator` for asynchronous streaming) that yields a single value: the final output from the underlying chat model provider.

The ability to stream the output token-by-token depends on whether the provider has implemented proper streaming support.

See which [integrations support token-by-token streaming here](/docs/integrations/llms/).



:::note

The **default** implementation does **not** provide support for token-by-token streaming, but it ensures that the model can be swapped in for any other model as it supports the same standard interface.

:::

## Sync stream

Below we use a `|` to help visualize the delimiter between tokens.
"""
logger.info("# How to stream responses from an LLM")


llm = ChatOllama(model="llama3.2", temperature=0, max_tokens=512)
for chunk in llm.stream("Write me a 1 verse song about sparkling water."):
    logger.debug(chunk, end="|", flush=True)

"""
## Async streaming

Let's see how to stream in an async setting using `astream`.
"""
logger.info("## Async streaming")


llm = ChatOllama(model="llama3.2", temperature=0, max_tokens=512)
for chunk in llm.stream("Write me a 1 verse song about sparkling water."):
    logger.debug(chunk, end="|", flush=True)

"""
## Async event streaming


LLMs also support the standard [astream events](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) method.

:::tip

`astream_events` is most useful when implementing streaming in a larger LLM application that contains multiple steps (e.g., an application that involves an `agent`).
:::
"""
logger.info("## Async event streaming")


llm = ChatOllama(model="llama3.2", temperature=0, max_tokens=512)

idx = 0

for event in llm.stream_events(
    "Write me a 1 verse song about goldfish on the moon", version="v1"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        logger.debug("...Truncated")
        break
    logger.debug(event)

logger.info("\n\n[DONE]", bright=True)
