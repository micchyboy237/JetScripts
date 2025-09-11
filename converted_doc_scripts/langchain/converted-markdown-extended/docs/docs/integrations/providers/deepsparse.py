from jet.logger import logger
from langchain_community.llms import DeepSparse
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
# DeepSparse

This page covers how to use the [DeepSparse](https://github.com/neuralmagic/deepsparse) inference runtime within LangChain.
It is broken into two parts: installation and setup, and then examples of DeepSparse usage.

## Installation and Setup

- Install the Python package with `pip install deepsparse`
- Choose a [SparseZoo model](https://sparsezoo.neuralmagic.com/?useCase=text_generation) or export a support model to ONNX [using Optimum](https://github.com/neuralmagic/notebooks/blob/main/notebooks/opt-text-generation-deepsparse-quickstart/OPT_Text_Generation_DeepSparse_Quickstart.ipynb)


## LLMs

There exists a DeepSparse LLM wrapper, which you can access with:
"""
logger.info("# DeepSparse")


"""
It provides a unified interface for all models:
"""
logger.info("It provides a unified interface for all models:")

llm = DeepSparse(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none')

logger.debug(llm.invoke('def fib():'))

"""
Additional parameters can be passed using the `config` parameter:
"""
logger.info("Additional parameters can be passed using the `config` parameter:")

config = {'max_generated_tokens': 256}

llm = DeepSparse(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none', config=config)

logger.info("\n\n[DONE]", bright=True)