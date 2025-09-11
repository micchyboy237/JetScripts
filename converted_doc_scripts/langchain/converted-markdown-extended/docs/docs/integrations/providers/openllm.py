from jet.logger import logger
from langchain_community.llms import OpenLLM
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
---
keywords: [openllm]
---

# OpenLLM

OpenLLM lets developers run any **open-source LLMs** as **Ollama-compatible API** endpoints with **a single command**.

- 🔬 Build for fast and production usages
- 🚂 Support llama3, qwen2, gemma, etc, and many **quantized** versions [full list](https://github.com/bentoml/openllm-models)
- ⛓️ Ollama-compatible API
- 💬 Built-in ChatGPT like UI
- 🔥 Accelerated LLM decoding with state-of-the-art inference backends
- 🌥️ Ready for enterprise-grade cloud deployment (Kubernetes, Docker and BentoCloud)

## Installation and Setup

Install the OpenLLM package via PyPI:
"""
logger.info("# OpenLLM")

pip install openllm

"""
## LLM

OpenLLM supports a wide range of open-source LLMs as well as serving users' own
fine-tuned LLMs. Use `openllm model` command to see all available models that
are pre-optimized for OpenLLM.

## Wrappers

There is a OpenLLM Wrapper which supports interacting with running server with OpenLLM:
"""
logger.info("## LLM")


"""
### Wrapper for OpenLLM server

This wrapper supports interacting with OpenLLM's Ollama-compatible endpoint.

To run a model, do:
"""
logger.info("### Wrapper for OpenLLM server")

openllm hello

"""
Wrapper usage:
"""
logger.info("Wrapper usage:")


llm = OpenLLM(base_url="http://localhost:3000/v1")

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")

"""
### Usage

For a more detailed walkthrough of the OpenLLM Wrapper, see the
[example notebook](/docs/integrations/llms/openllm)
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)