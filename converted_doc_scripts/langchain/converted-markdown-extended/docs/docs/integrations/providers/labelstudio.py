from jet.logger import logger
from langchain.callbacks import LabelStudioCallbackHandler
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
# Label Studio


>[Label Studio](https://labelstud.io/guide/get_started) is an open-source data labeling platform that provides LangChain with flexibility when it comes to labeling data for fine-tuning large language models (LLMs). It also enables the preparation of custom training data and the collection and evaluation of responses through human feedback.

## Installation and Setup

See the [Label Studio installation guide](https://labelstud.io/guide/install) for installation options.

We need to install the  `label-studio` and `label-studio-sdk-python` Python packages:
"""
logger.info("# Label Studio")

pip install label-studio label-studio-sdk

"""
## Callbacks

See a [usage example](/docs/integrations/callbacks/labelstudio).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)