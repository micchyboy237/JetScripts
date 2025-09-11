from jet.logger import logger
from langchain_text_splitters import KonlpyTextSplitter
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
# KoNLPY

>[KoNLPy](https://konlpy.org/) is a Python package for natural language processing (NLP)
> of the Korean language.


## Installation and Setup

You need to install the `konlpy` python package.
"""
logger.info("# KoNLPY")

pip install konlpy

"""
## Text splitter

See a [usage example](/docs/how_to/split_by_token/#konlpy).
"""
logger.info("## Text splitter")


logger.info("\n\n[DONE]", bright=True)