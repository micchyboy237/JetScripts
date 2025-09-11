from jet.logger import logger
from langchain_community.llms import CTranslate2
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
# CTranslate2

>[CTranslate2](https://opennmt.net/CTranslate2/quickstart.html) is a C++ and Python library
> for efficient inference with Transformer models.
>
>The project implements a custom runtime that applies many performance optimization
> techniques such as weights quantization, layers fusion, batch reordering, etc.,
> to accelerate and reduce the memory usage of Transformer models on CPU and GPU.
>
>A full list of features and supported models is included in the
> [project’s repository](https://opennmt.net/CTranslate2/guides/transformers.html).
> To start, please check out the official [quickstart guide](https://opennmt.net/CTranslate2/quickstart.html).


## Installation and Setup

Install the Python package:
"""
logger.info("# CTranslate2")

pip install ctranslate2

"""
## LLMs

See a [usage example](/docs/integrations/llms/ctranslate2).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)