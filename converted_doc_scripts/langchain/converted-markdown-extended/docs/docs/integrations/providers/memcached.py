from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain_community.cache import MemcachedCache
from pymemcache.client.base import Client
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
# Memcached

> [Memcached](https://www.memcached.org/) is a free & open source, high-performance, distributed memory object caching system,
> generic in nature, but intended for use in speeding up dynamic web applications by alleviating database load.

This page covers how to use Memcached with langchain, using [pymemcache](https://github.com/pinterest/pymemcache) as
a client to connect to an already running Memcached instance.

## Installation and Setup
"""
logger.info("# Memcached")

pip install pymemcache

"""
## LLM Cache

To integrate a Memcached Cache into your application:
"""
logger.info("## LLM Cache")



llm = Ollama(model="llama3.2", n=2, best_of=2)
set_llm_cache(MemcachedCache(Client('localhost')))

llm.invoke("Which city is the most crowded city in the USA?")

llm.invoke("Which city is the most crowded city in the USA?")

"""
Learn more in the [example notebook](/docs/integrations/llm_caching#memcached-cache)
"""
logger.info("Learn more in the [example notebook](/docs/integrations/llm_caching#memcached-cache)")

logger.info("\n\n[DONE]", bright=True)