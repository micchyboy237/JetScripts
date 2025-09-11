from jet.logger import logger
from langchain_fireworks import ChatFireworks
from langchain_fireworks import Fireworks
from langchain_fireworks import FireworksEmbeddings
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
# Fireworks AI

>[Fireworks AI](https://fireworks.ai) is a generative AI inference platform to run and
> customize models with industry-leading speed and production-readiness.



## Installation and setup

- Install the Fireworks integration package.

  ```bash
#   pip install langchain-fireworks
  ```

- Get a Fireworks API key by signing up at [fireworks.ai](https://fireworks.ai).
- Authenticate by setting the FIREWORKS_API_KEY environment variable.

### Authentication

There are two ways to authenticate using your Fireworks API key:

1.  Setting the `FIREWORKS_API_KEY` environment variable.

    ```python
    os.environ["FIREWORKS_API_KEY"] = "<KEY>"
    ```

2.  Setting `api_key` field in the Fireworks LLM module.

    ```python
    llm = Fireworks()
    ```
## Chat models

See a [usage example](/docs/integrations/chat/fireworks).
"""
logger.info("# Fireworks AI")


"""
## LLMs

See a [usage example](/docs/integrations/llms/fireworks).
"""
logger.info("## LLMs")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/fireworks).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)