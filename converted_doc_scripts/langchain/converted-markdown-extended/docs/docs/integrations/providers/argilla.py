from jet.logger import logger
from langchain.callbacks import ArgillaCallbackHandler
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
# Argilla

>[Argilla](https://argilla.io/) is an open-source data curation platform for LLMs.
> Using `Argilla`, everyone can build robust language models through faster data curation
> using both human and machine feedback. `Argilla` provides support for each step in the MLOps cycle,
> from data labeling to model monitoring.

## Installation and Setup

Get your [API key](https://platform.ollama.com/account/api-keys).

Install the Python package:
"""
logger.info("# Argilla")

pip install argilla

"""
## Callbacks
"""
logger.info("## Callbacks")


"""
See an [example](/docs/integrations/callbacks/argilla).
"""
logger.info("See an [example](/docs/integrations/callbacks/argilla).")

logger.info("\n\n[DONE]", bright=True)