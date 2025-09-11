from jet.logger import logger
from langchain_community.llms import CTransformers
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
# C Transformers

This page covers how to use the [C Transformers](https://github.com/marella/ctransformers) library within LangChain.
It is broken into two parts: installation and setup, and then references to specific C Transformers wrappers.

## Installation and Setup

- Install the Python package with `pip install ctransformers`
- Download a supported [GGML model](https://huggingface.co/TheBloke) (see [Supported Models](https://github.com/marella/ctransformers#supported-models))

## Wrappers

### LLM

There exists a CTransformers LLM wrapper, which you can access with:
"""
logger.info("# C Transformers")


"""
It provides a unified interface for all models:
"""
logger.info("It provides a unified interface for all models:")

llm = CTransformers(model='/path/to/ggml-gpt-2.bin', model_type='gpt2')

logger.debug(llm.invoke('AI is going to'))

"""
If you are getting `illegal instruction` error, try using `lib='avx'` or `lib='basic'`:
"""
logger.info("If you are getting `illegal instruction` error, try using `lib='avx'` or `lib='basic'`:")

llm = CTransformers(model='/path/to/ggml-gpt-2.bin', model_type='gpt2', lib='avx')

"""
It can be used with models hosted on the Hugging Face Hub:
"""
logger.info("It can be used with models hosted on the Hugging Face Hub:")

llm = CTransformers(model='marella/gpt-2-ggml')

"""
If a model repo has multiple model files (`.bin` files), specify a model file using:
"""
logger.info("If a model repo has multiple model files (`.bin` files), specify a model file using:")

llm = CTransformers(model='marella/gpt-2-ggml', model_file='ggml-model.bin')

"""
Additional parameters can be passed using the `config` parameter:
"""
logger.info("Additional parameters can be passed using the `config` parameter:")

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1}

llm = CTransformers(model='marella/gpt-2-ggml', config=config)

"""
See [Documentation](https://github.com/marella/ctransformers#config) for a list of available parameters.

For a more detailed walkthrough of this, see [this notebook](/docs/integrations/llms/ctransformers).
"""
logger.info("See [Documentation](https://github.com/marella/ctransformers#config) for a list of available parameters.")

logger.info("\n\n[DONE]", bright=True)