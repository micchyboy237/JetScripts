from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
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
# How to cache LLM responses

LangChain provides an optional [caching](/docs/concepts/chat_models/#caching) layer for LLMs. This is useful for two reasons:

It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times.
It can speed up your application by reducing the number of API calls you make to the LLM provider.
"""
logger.info("# How to cache LLM responses")

# %pip install -qU jet.adapters.langchain.chat_ollama langchain_community

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


llm = ChatOllama(model="llama3.2", n=2, best_of=2)

# %%time

set_llm_cache(InMemoryCache())

llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## SQLite Cache
"""
logger.info("## SQLite Cache")

# !rm .langchain.db


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

logger.info("\n\n[DONE]", bright=True)
