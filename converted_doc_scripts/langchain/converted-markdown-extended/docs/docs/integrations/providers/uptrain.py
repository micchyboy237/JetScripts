from jet.logger import logger
from langchain_community.callbacks.uptrain_callback import UpTrainCallbackHandler
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
# UpTrain

>[UpTrain](https://uptrain.ai/) is an open-source unified platform to evaluate and
>improve Generative AI applications. It provides grades for 20+ preconfigured evaluations
>(covering language, code, embedding use cases), performs root cause analysis on failure
>cases and gives insights on how to resolve them.

## Installation and Setup
"""
logger.info("# UpTrain")

pip install uptrain

"""
## Callbacks
"""
logger.info("## Callbacks")


"""
See an [example](/docs/integrations/callbacks/uptrain).
"""
logger.info("See an [example](/docs/integrations/callbacks/uptrain).")

logger.info("\n\n[DONE]", bright=True)