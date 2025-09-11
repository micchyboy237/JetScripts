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
# Deeplake

[Deeplake](https://www.deeplake.ai/) is a database optimized for AI and deep learning
applications.


## Installation and Setup
"""
logger.info("# Deeplake")

pip install langchain-deeplake

"""
## Vector stores

See detail on available vector stores
[here](/docs/integrations/vectorstores/activeloop_deeplake).
"""
logger.info("## Vector stores")

logger.info("\n\n[DONE]", bright=True)