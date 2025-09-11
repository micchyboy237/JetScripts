from jet.logger import logger
from langchain.callbacks.confident_callback import DeepEvalCallbackHandler
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
# Confident AI

>[Confident AI](https://confident-ai.com) is a creator of the `DeepEval`.
>
>[DeepEval](https://github.com/confident-ai/deepeval) is a package for unit testing LLMs.
> Using `DeepEval`, everyone can build robust language models through faster iterations
> using both unit testing and integration testing. `DeepEval provides support for each step in the iteration
> from synthetic data creation to testing.

## Installation and Setup

You need to get the [DeepEval API credentials](https://app.confident-ai.com).

You need to install the `DeepEval` Python package:
"""
logger.info("# Confident AI")

pip install deepeval

"""
## Callbacks

See an [example](/docs/integrations/callbacks/confident).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)