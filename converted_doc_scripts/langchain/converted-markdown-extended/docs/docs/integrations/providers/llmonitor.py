from jet.logger import logger
from langchain.callbacks import LLMonitorCallbackHandler
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
# LLMonitor

>[LLMonitor](https://llmonitor.com?utm_source=langchain&utm_medium=py&utm_campaign=docs) is an open-source observability platform that provides cost and usage analytics, user tracking, tracing and evaluation tools.

## Installation and Setup

Create an account on [llmonitor.com](https://llmonitor.com?utm_source=langchain&utm_medium=py&utm_campaign=docs), then copy your new app's `tracking id`.

Once you have it, set it as an environment variable by running:
"""
logger.info("# LLMonitor")

export LLMONITOR_APP_ID="..."

"""
## Callbacks

See a [usage example](/docs/integrations/callbacks/llmonitor).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)