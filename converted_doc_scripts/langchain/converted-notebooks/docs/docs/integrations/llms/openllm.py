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
# OpenLLM

[🦾 OpenLLM](https://github.com/bentoml/OpenLLM) lets developers run any **open-source LLMs** as **Ollama-compatible API** endpoints with **a single command**.

- 🔬 Build for fast and production usages
- 🚂 Support llama3, qwen2, gemma, etc, and many **quantized** versions [full list](https://github.com/bentoml/openllm-models)
- ⛓️ Ollama-compatible API
- 💬 Built-in ChatGPT like UI
- 🔥 Accelerated LLM decoding with state-of-the-art inference backends
- 🌥️ Ready for enterprise-grade cloud deployment (Kubernetes, Docker and BentoCloud)

## Installation

Install `openllm` through [PyPI](https://pypi.org/project/openllm/)
"""
logger.info("# OpenLLM")

# %pip install --upgrade --quiet  openllm

"""
## Launch OpenLLM server locally

To start an LLM server, use `openllm hello` command:

```bash
openllm hello
```


## Wrapper
"""
logger.info("## Launch OpenLLM server locally")


server_url = "http://localhost:3000"  # Replace with remote host if you are running on a remote server
llm = OpenLLM(base_url=server_url)

llm("To build a LLM from scratch, the following are the steps:")

logger.info("\n\n[DONE]", bright=True)