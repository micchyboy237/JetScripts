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
# Remembrall

>[Remembrall](https://remembrall.dev/) is a platform that gives a language model
> long-term memory, retrieval augmented generation, and complete observability.

## Installation and Setup

To get started, [sign in with Github on the Remembrall platform](https://remembrall.dev/login)
and copy your [API key from the settings page](https://remembrall.dev/dashboard/settings).


## Memory

See a [usage example](/docs/integrations/memory/remembrall).
"""
logger.info("# Remembrall")

logger.info("\n\n[DONE]", bright=True)