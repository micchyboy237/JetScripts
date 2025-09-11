from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os
import shutil
import sqlalchemy


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
# Motherduck

>[Motherduck](https://motherduck.com/) is a managed DuckDB-in-the-cloud service.

## Installation and Setup

First, you need to install `duckdb` python package.
"""
logger.info("# Motherduck")

pip install duckdb

"""
You will also need to sign up for an account at [Motherduck](https://motherduck.com/)

After that, you should set up a connection string - we mostly integrate with Motherduck through SQLAlchemy.
The connection string is likely in the form:

token="..."

conn_str = f"duckdb:///md:{token}@my_db"

## SQLChain

You can use the SQLChain to query data in your Motherduck instance in natural language.

db = SQLDatabase.from_uri(conn_str)
db_chain = SQLDatabaseChain.from_llm(Ollama(temperature=0), db, verbose=True)

From here, see the [SQL Chain](/docs/how_to#qa-over-sql--csv) documentation on how to use.


## LLMCache

You can also easily use Motherduck to cache LLM requests.
Once again this is done through the SQLAlchemy wrapper.

eng = sqlalchemy.create_engine(conn_str)
set_llm_cache(SQLAlchemyCache(engine=eng))

From here, see the [LLM Caching](/docs/integrations/llm_caching) documentation on how to use.
"""
logger.info("## SQLChain")

logger.info("\n\n[DONE]", bright=True)