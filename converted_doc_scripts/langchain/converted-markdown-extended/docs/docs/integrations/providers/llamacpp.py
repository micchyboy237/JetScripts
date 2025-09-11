from jet.logger import logger
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
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
# Llama.cpp

>[llama.cpp python](https://github.com/abetlen/llama-cpp-python) library is a simple Python bindings for `@ggerganov`
>[llama.cpp](https://github.com/ggerganov/llama.cpp).
>
>This package provides:
>
> - Low-level access to C API via ctypes interface.
> - High-level Python API for text completion
>   - `Ollama`-like API
>   - `LangChain` compatibility
>   - `LlamaIndex` compatibility
> - Ollama compatible web server
>   - Local Copilot replacement
>   - Function Calling support
>   - Vision API support
>   - Multiple Models

## Installation and Setup

- Install the Python package
  ```bash
#   pip install llama-cpp-python
  ````
- Download one of the [supported models](https://github.com/ggerganov/llama.cpp#description) and convert them to the llama.cpp format per the [instructions](https://github.com/ggerganov/llama.cpp)


## Chat models

See a [usage example](/docs/integrations/chat/llamacpp).
"""
logger.info("# Llama.cpp")


"""
## LLMs

See a [usage example](/docs/integrations/llms/llamacpp).
"""
logger.info("## LLMs")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/llamacpp).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)