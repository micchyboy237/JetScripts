from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
import ChatModelTabs from "@theme/ChatModelTabs";
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
# How to cache chat model responses

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [LLMs](/docs/concepts/text_llms)

:::

LangChain provides an optional caching layer for [chat models](/docs/concepts/chat_models). This is useful for two main reasons:

- It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times. This is especially useful during app development.
- It can speed up your application by reducing the number of API calls you make to the LLM provider.

This guide will walk you through how to enable this in your apps.


<ChatModelTabs customVarName="llm" />
"""
logger.info("# How to cache chat model responses")

# from getpass import getpass


# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOllama(model="llama3.2")


"""
## In Memory Cache

This is an ephemeral cache that stores model calls in memory. It will be wiped when your environment restarts, and is not shared across processes.
"""
logger.info("## In Memory Cache")

# %%time

set_llm_cache(InMemoryCache())

llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## SQLite Cache

This cache implementation uses a `SQLite` database to store responses, and will last across process restarts.
"""
logger.info("## SQLite Cache")

# !rm .langchain.db


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## Next steps

You've now learned how to cache model responses to save time and money.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to create your own custom chat model](/docs/how_to/custom_chat_model).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)