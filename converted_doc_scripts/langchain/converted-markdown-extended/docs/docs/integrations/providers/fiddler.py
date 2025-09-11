from jet.logger import logger
from langchain_community.callbacks.fiddler_callback import FiddlerCallbackHandler
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
# Fiddler

>[Fiddler](https://www.fiddler.ai/) provides a unified platform to monitor, explain, analyze,
> and improve ML deployments at an enterprise scale.

## Installation and Setup

Set up your model [with Fiddler](https://demo.fiddler.ai):

* The URL you're using to connect to Fiddler
* Your organization ID
* Your authorization token

Install the Python package:
"""
logger.info("# Fiddler")

pip install fiddler-client

"""
## Callbacks
"""
logger.info("## Callbacks")


"""
See an [example](/docs/integrations/callbacks/fiddler).
"""
logger.info("See an [example](/docs/integrations/callbacks/fiddler).")

logger.info("\n\n[DONE]", bright=True)