from jet.logger import logger
from langchain_predictionguard import ChatPredictionGuard
from langchain_predictionguard import PredictionGuard
from langchain_predictionguard import PredictionGuardEmbeddings
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
# Prediction Guard

This page covers how to use the Prediction Guard ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific Prediction Guard wrappers.

This integration is maintained in the [langchain-predictionguard](https://github.com/predictionguard/langchain-predictionguard)
package.

## Installation and Setup

- Install the PredictionGuard Langchain partner package:

# pip install langchain-predictionguard

- Get a Prediction Guard API key (as described [here](https://docs.predictionguard.com/)) and set it as an environment variable (`PREDICTIONGUARD_API_KEY`)

## Prediction Guard Langchain Integrations
|API|Description|Endpoint Docs| Import                                                  | Example Usage                                                                 |
|---|---|---|---------------------------------------------------------|-------------------------------------------------------------------------------|
|Chat|Build Chat Bots|[Chat](https://docs.predictionguard.com/api-reference/api-reference/chat-completions)| `from langchain_predictionguard import ChatPredictionGuard` | [ChatPredictionGuard.ipynb](/docs/integrations/chat/predictionguard)             |
|Completions|Generate Text|[Completions](https://docs.predictionguard.com/api-reference/api-reference/completions)| `from langchain_predictionguard import PredictionGuard` | [PredictionGuard.ipynb](/docs/integrations/llms/predictionguard)                     |
|Text Embedding|Embed String to Vectors|[Embeddings](https://docs.predictionguard.com/api-reference/api-reference/embeddings)| `from langchain_predictionguard import PredictionGuardEmbeddings` | [PredictionGuardEmbeddings.ipynb](/docs/integrations/text_embedding/predictionguard) |

## Getting Started

## Chat Models

### Prediction Guard Chat

See a [usage example](/docs/integrations/chat/predictionguard)
"""
logger.info("# Prediction Guard")


"""
#### Usage
"""
logger.info("#### Usage")

chat = ChatPredictionGuard(model="Hermes-3-Llama-3.1-8B")

chat.invoke("Tell me a joke")

"""
## Embedding Models

### Prediction Guard Embeddings

See a [usage example](/docs/integrations/text_embedding/predictionguard)
"""
logger.info("## Embedding Models")


"""
#### Usage
"""
logger.info("#### Usage")

embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")

text = "This is an embedding example."
output = embeddings.embed_query(text)

"""
## LLMs

### Prediction Guard LLM

See a [usage example](/docs/integrations/llms/predictionguard)
"""
logger.info("## LLMs")


"""
#### Usage
"""
logger.info("#### Usage")

llm = PredictionGuard(model="Hermes-2-Pro-Llama-3-8B")

llm.invoke("Tell me a joke about bears")

logger.info("\n\n[DONE]", bright=True)