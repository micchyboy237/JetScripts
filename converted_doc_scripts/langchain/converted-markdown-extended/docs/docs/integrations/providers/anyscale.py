from jet.logger import logger
from langchain_community.chat_models.anyscale import ChatAnyscale
from langchain_community.embeddings import AnyscaleEmbeddings
from langchain_community.llms.anyscale import Anyscale
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
# Anyscale

>[Anyscale](https://www.anyscale.com) is a platform to run, fine tune and scale LLMs via production-ready APIs.
> [Anyscale Endpoints](https://docs.anyscale.com/endpoints/overview) serve many open-source models in a cost-effective way.

`Anyscale` also provides [an example](https://docs.anyscale.com/endpoints/model-serving/examples/langchain-integration)
how to setup `LangChain` with `Anyscale` for advanced chat agents.

## Installation and Setup

- Get an Anyscale Service URL, route and API key and set them as environment variables (`ANYSCALE_SERVICE_URL`,`ANYSCALE_SERVICE_ROUTE`, `ANYSCALE_SERVICE_TOKEN`).
- Please see [the Anyscale docs](https://www.anyscale.com/get-started) for more details.

We have to install the `ollama` package:
"""
logger.info("# Anyscale")

pip install ollama

"""
## LLM

See a [usage example](/docs/integrations/llms/anyscale).
"""
logger.info("## LLM")


"""
## Chat Models

See a [usage example](/docs/integrations/chat/anyscale).
"""
logger.info("## Chat Models")


"""
## Embeddings

See a [usage example](/docs/integrations/text_embedding/anyscale).
"""
logger.info("## Embeddings")


logger.info("\n\n[DONE]", bright=True)