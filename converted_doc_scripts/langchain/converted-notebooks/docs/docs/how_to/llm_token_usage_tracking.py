from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.callbacks import get_ollama_callback
from langchain_core.prompts import PromptTemplate
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
# How to track token usage for LLMs

Tracking [token](/docs/concepts/tokens/) usage to calculate cost is an important part of putting your app in production. This guide goes over how to obtain this information from your LangChain model calls.

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [LLMs](/docs/concepts/text_llms)
:::

## Using LangSmith

You can use [LangSmith](https://www.langchain.com/langsmith) to help track token usage in your LLM application. See the [LangSmith quick start guide](https://docs.smith.langchain.com/).

## Using callbacks

There are some API-specific callback context managers that allow you to track token usage across multiple calls. You'll need to check whether such an integration is available for your particular model.

If such an integration is not available for your model, you can create a custom callback manager by adapting the implementation of the [Ollama callback manager](https://python.langchain.com/api_reference/community/callbacks/langchain_community.callbacks.ollama_info.OllamaCallbackHandler.html).

### Ollama

Let's first look at an extremely simple example of tracking token usage for a single Chat model call.

:::danger

The callback handler does not currently support streaming token counts for legacy language models (e.g., `jet.adapters.langchain.chat_ollama.Ollama`). For support in a streaming context, refer to the corresponding guide for chat models [here](/docs/how_to/chat_token_usage_tracking).

:::

### Single call
"""
logger.info("# How to track token usage for LLMs")


llm = ChatOllama(model_name="gpt-3.5-turbo-instruct")

with get_ollama_callback() as cb:
    result = llm.invoke("Tell me a joke")
    logger.debug(result)
    logger.debug("---")
logger.debug()

logger.debug(f"Total Tokens: {cb.total_tokens}")
logger.debug(f"Prompt Tokens: {cb.prompt_tokens}")
logger.debug(f"Completion Tokens: {cb.completion_tokens}")
logger.debug(f"Total Cost (USD): ${cb.total_cost}")

"""
### Multiple calls

Anything inside the context manager will get tracked. Here's an example of using it to track multiple calls in sequence to a chain. This will also work for an agent which may use multiple steps.
"""
logger.info("### Multiple calls")


llm = ChatOllama(model_name="gpt-3.5-turbo-instruct")

template = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = template | llm

with get_ollama_callback() as cb:
    response = chain.invoke({"topic": "birds"})
    logger.debug(response)
    response = chain.invoke({"topic": "fish"})
    logger.debug("--")
    logger.debug(response)


logger.debug()
logger.debug("---")
logger.debug(f"Total Tokens: {cb.total_tokens}")
logger.debug(f"Prompt Tokens: {cb.prompt_tokens}")
logger.debug(f"Completion Tokens: {cb.completion_tokens}")
logger.debug(f"Total Cost (USD): ${cb.total_cost}")

"""
## Streaming

:::danger

`get_ollama_callback` does not currently support streaming token counts for legacy language models (e.g., `jet.adapters.langchain.chat_ollama.Ollama`). If you want to count tokens correctly in a streaming context, there are a number of options:

- Use chat models as described in [this guide](/docs/how_to/chat_token_usage_tracking);
- Implement a [custom callback handler](/docs/how_to/custom_callbacks/) that uses appropriate tokenizers to count the tokens;
- Use a monitoring platform such as [LangSmith](https://www.langchain.com/langsmith).
:::

Note that when using legacy language models in a streaming context, token counts are not updated:
"""
logger.info("## Streaming")


llm = ChatOllama(model_name="gpt-3.5-turbo-instruct")

with get_ollama_callback() as cb:
    for chunk in llm.stream("Tell me a joke"):
        logger.debug(chunk, end="", flush=True)
    logger.debug(result)
    logger.debug("---")
logger.debug()

logger.debug(f"Total Tokens: {cb.total_tokens}")
logger.debug(f"Prompt Tokens: {cb.prompt_tokens}")
logger.debug(f"Completion Tokens: {cb.completion_tokens}")
logger.debug(f"Total Cost (USD): ${cb.total_cost}")

logger.info("\n\n[DONE]", bright=True)
