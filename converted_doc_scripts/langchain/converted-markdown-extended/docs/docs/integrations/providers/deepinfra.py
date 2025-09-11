from jet.logger import logger
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.llms import DeepInfra
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
# DeepInfra

>[DeepInfra](https://deepinfra.com/docs) allows us to run the
> [latest machine learning models](https://deepinfra.com/models) with ease.
> DeepInfra takes care of all the heavy lifting related to running, scaling and monitoring
> the models. Users can focus on your application and integrate the models with simple REST API calls.

>DeepInfra provides [examples](https://deepinfra.com/docs/advanced/langchain) of integration with LangChain.

This page covers how to use the `DeepInfra` ecosystem within `LangChain`.
It is broken into two parts: installation and setup, and then references to specific DeepInfra wrappers.

## Installation and Setup

- Get your DeepInfra api key from this link [here](https://deepinfra.com/).
- Get an DeepInfra api key and set it as an environment variable (`DEEPINFRA_API_TOKEN`)

## Available Models

DeepInfra provides a range of Open Source LLMs ready for deployment.

You can see supported models for
[text-generation](https://deepinfra.com/models?type=text-generation) and
[embeddings](https://deepinfra.com/models?type=embeddings).

You can view a [list of request and response parameters](https://deepinfra.com/meta-llama/Llama-2-70b-chat-hf/api).

Chat models [follow ollama api](https://deepinfra.com/meta-llama/Llama-2-70b-chat-hf/api?example=ollama-http)


## LLM

See a [usage example](/docs/integrations/llms/deepinfra).
"""
logger.info("# DeepInfra")


"""
## Embeddings

See a [usage example](/docs/integrations/text_embedding/deepinfra).
"""
logger.info("## Embeddings")


"""
## Chat Models

See a [usage example](/docs/integrations/chat/deepinfra).
"""
logger.info("## Chat Models")


logger.info("\n\n[DONE]", bright=True)