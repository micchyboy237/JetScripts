from jet.logger import logger
from langchain.callbacks import ContextCallbackHandler
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
# Context

>[Context](https://context.ai/) provides user analytics for LLM-powered products and features.

## Installation and Setup

We need to install the  `context-python` Python package:
"""
logger.info("# Context")

pip install context-python

"""
## Callbacks

See a [usage example](/docs/integrations/callbacks/context).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)