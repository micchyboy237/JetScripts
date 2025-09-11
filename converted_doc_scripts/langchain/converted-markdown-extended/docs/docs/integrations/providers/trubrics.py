from jet.logger import logger
from langchain.callbacks import TrubricsCallbackHandler
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
# Trubrics


>[Trubrics](https://trubrics.com) is an LLM user analytics platform that lets you collect, analyse and manage user
prompts & feedback on AI models.
>
>Check out [Trubrics repo](https://github.com/trubrics/trubrics-sdk) for more information on `Trubrics`.

## Installation and Setup

We need to install the  `trubrics` Python package:
"""
logger.info("# Trubrics")

pip install trubrics

"""
## Callbacks

See a [usage example](/docs/integrations/callbacks/trubrics).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)