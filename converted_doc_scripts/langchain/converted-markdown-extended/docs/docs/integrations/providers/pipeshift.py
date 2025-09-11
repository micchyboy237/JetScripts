from jet.logger import logger
from langchain_pipeshift import ChatPipeshift
from langchain_pipeshift import Pipeshift
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
# Pipeshift

> [Pipeshift](https://pipeshift.com) is a fine-tuning and inference platform for open-source LLMs

- You bring your datasets. Fine-tune multiple LLMs. Start inferencing in one-click and watch them scale to millions.

## Installation and Setup

- Install the Pipeshift integration package.

  ```bash
#   pip install langchain-pipeshift
  ```

- Get your Pipeshift API key by signing up at [Pipeshift](https://pipeshift.com).

### Authentication

You can perform authentication using your Pipeshift API key in any of the following ways:

1. Adding API key to the environment variable as `PIPESHIFT_API_KEY`.

    ```python
    os.environ["PIPESHIFT_API_KEY"] = "<your_api_key>"
    ```

2. By passing `api_key` to the pipeshift LLM module or chat module

    ```python
    llm = Pipeshift(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=512)

                        OR

    chat = ChatPipeshift(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=512)
    ```

## Chat models

See an [example](/docs/integrations/chat/pipeshift).
"""
logger.info("# Pipeshift")


"""
## LLMs

See an [example](/docs/integrations/llms/pipeshift).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)