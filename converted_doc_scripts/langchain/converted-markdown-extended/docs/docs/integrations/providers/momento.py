from datetime import timedelta
from jet.logger import logger
from langchain.cache import MomentoCache
from langchain.globals import set_llm_cache
from langchain.memory import MomentoChatMessageHistory
from langchain_community.vectorstores import MomentoVectorIndex
from momento import CacheClient, Configurations, CredentialProvider
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
# Momento

> [Momento Cache](https://docs.momentohq.com/) is the world's first truly serverless caching service, offering instant elasticity, scale-to-zero
> capability, and blazing-fast performance.
>
> [Momento Vector Index](https://docs.momentohq.com/vector-index) stands out as the most productive, easiest-to-use, fully serverless vector index.
>
> For both services, simply grab the SDK, obtain an API key, input a few lines into your code, and you're set to go. Together, they provide a comprehensive solution for your LLM data needs.

This page covers how to use the [Momento](https://gomomento.com) ecosystem within LangChain.

## Installation and Setup

- Sign up for a free account [here](https://console.gomomento.com/) to get an API key
- Install the Momento Python SDK with `pip install momento`

## Cache

Use Momento as a serverless, distributed, low-latency cache for LLM prompts and responses. The standard cache is the primary use case for Momento users in any environment.

To integrate Momento Cache into your application:
"""
logger.info("# Momento")


"""
Then, set it up with the following code:
"""
logger.info("Then, set it up with the following code:")


cache_client = CacheClient(
    Configurations.Laptop.v1(),
    CredentialProvider.from_environment_variable("MOMENTO_API_KEY"),
    default_ttl=timedelta(days=1))

cache_name = "langchain"

set_llm_cache(MomentoCache(cache_client, cache_name))

"""
## Memory

Momento can be used as a distributed memory store for LLMs.

See [this notebook](/docs/integrations/memory/momento_chat_message_history) for a walkthrough of how to use Momento as a memory store for chat message history.
"""
logger.info("## Memory")


"""
## Vector Store

Momento Vector Index (MVI) can be used as a vector store.

See [this notebook](/docs/integrations/vectorstores/momento_vector_index) for a walkthrough of how to use MVI as a vector store.
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)