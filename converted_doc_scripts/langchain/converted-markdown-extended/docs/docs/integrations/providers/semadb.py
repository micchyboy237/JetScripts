from jet.logger import logger
from langchain_community.vectorstores import SemaDB
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
# SemaDB

>[SemaDB](https://semafind.com/) is a no fuss vector similarity search engine. It provides a low-cost cloud hosted version to help you build AI applications with ease.

With SemaDB Cloud, our hosted version, no fuss means no pod size calculations, no schema definitions, no partition settings, no parameter tuning, no search algorithm tuning, no complex installation, no complex API. It is integrated with [RapidAPI](https://rapidapi.com/semafind-semadb/api/semadb) providing transparent billing, automatic sharding and an interactive API playground.

## Installation

None required, get started directly with SemaDB Cloud at [RapidAPI](https://rapidapi.com/semafind-semadb/api/semadb).

## Vector Store

There is a basic wrapper around `SemaDB` collections allowing you to use it as a vectorstore.
"""
logger.info("# SemaDB")


"""
You can follow a tutorial on how to use the wrapper in [this notebook](/docs/integrations/vectorstores/semadb).
"""
logger.info("You can follow a tutorial on how to use the wrapper in [this notebook](/docs/integrations/vectorstores/semadb).")

logger.info("\n\n[DONE]", bright=True)