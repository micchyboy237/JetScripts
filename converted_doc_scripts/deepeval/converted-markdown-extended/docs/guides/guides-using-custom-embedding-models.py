from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from jet.adapters.langchain.chat_ollama import AzureOpenAIEmbeddings
from jet.logger import logger
from typing import List, Optional
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
# id: using-custom-embedding-models
title: Using Custom Embedding Models
sidebar_label: Using Custom Embedding Models
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/guides/guides-using-custom-embedding-models"
  />
</head>

Throughout `deepeval`, only the `generate_goldens_from_docs()` method in the `Synthesizer` for synthetic data generation uses an embedding model. This is because in order to generate goldens from documents, the `Synthesizer` uses cosine similarity to generate the relevant context needed for data synthesization.

This guide will teach you how to use literally **ANY** embedding model to extract context from documents that are required for synthetic data generation.

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Using Azure Ollama

You can use Azure's Ollama embedding models by running the following commands in the CLI:
"""
logger.info("# id: using-custom-embedding-models")

deepeval set-azure-ollama \
    --ollama-endpoint=<endpoint> \
    --ollama-api-key=<api_key> \
    --ollama-model-name=<model_name> \
    --deployment-name=<deployment_name> \
    --ollama-api-version=<openai_api_version> \
    --model-version=<model_version> # e.g. 2024-11-20

"""
Then, run this to set the Azure Ollama embedder:
"""
logger.info("Then, run this to set the Azure Ollama embedder:")

deepeval set-azure-ollama-embedding --embedding-deployment-name=<embedding_deployment_name>

"""
:::tip Did You Know?
The first command configures `deepeval` to use Azure Ollama LLM globally, while the second command configures `deepeval` to use Azure Ollama's embedding models globally.
:::

### Using Ollama models

To use a local model served by Ollama, use the following command:
"""
logger.info("### Using Ollama models")

deepeval set-ollama <model_name>

"""
Where model_name is one of the LLM that appears when executing `ollama list`. If you ever wish to stop using your local LLM model and move back to regular Ollama, simply run:
"""
logger.info("Where model_name is one of the LLM that appears when executing `ollama list`. If you ever wish to stop using your local LLM model and move back to regular Ollama, simply run:")

deepeval unset-ollama

"""
Then, run this to set the local Embeddings model:
"""
logger.info("Then, run this to set the local Embeddings model:")

deepeval set-ollama-embeddings <embedding_model_name>

"""
To revert back to the default Ollama embeddings run:
"""
logger.info("To revert back to the default Ollama embeddings run:")

deepeval unset-ollama-embeddings

"""
### Using local LLM models

There are several local LLM providers that offer Ollama API compatible endpoints, like vLLM or LM Studio. You can use them with `deepeval` by setting several parameters from the CLI. To configure any of those providers, you need to supply the base URL where the service is running. These are some of the most popular alternatives for base URLs:

- LM Studio: `http://localhost:1234/v1/`
- vLLM: `http://localhost:8000/v1/`

For example to use a local model from LM Studio, use the following command:
"""
logger.info("### Using local LLM models")

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

"""
Then, run this to set the local Embeddings model:
"""
logger.info("Then, run this to set the local Embeddings model:")

deepeval set-local-embeddings --model-name=<embedding_model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

"""
To revert back to the default Ollama embeddings run:
"""
logger.info("To revert back to the default Ollama embeddings run:")

deepeval unset-local-embeddings

"""
For additional instructions about LLM model and embeddings model availability and base URLs, consult the provider's documentation.

### Using A Custom Embedding Model

Alternatively, you can also create a custom embedding model in code by inheriting the base `DeepEvalBaseEmbeddingModel` class. Here is an example of using the same custom Azure Ollama embedding model but created in code instead using langchain's `jet.adapters.langchain.chat_ollama` module:
"""
logger.info("### Using A Custom Embedding Model")


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return AzureOpenAIEmbeddings(
            openai_api_version="...",
            azure_deployment="...",
            azure_endpoint="...",
            openai_)

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        "Custom Azure Embedding Model"

"""
When creating a custom embedding model, you should **ALWAYS**:

- inherit `DeepEvalBaseEmbeddingModel`.
- implement the `get_model_name()` method, which simply returns a string representing your custom model name.
- implement the `load_model()` method, which will be responsible for returning the model object instance.
- implement the `embed_text()` method with **one and only one** parameter of type `str` as the text to be embedded, and returns a vector of type `List[float]`. We called `embedding_model.embed_query(prompt)` to access the embedded text in this particular example, but this could be different depending on the implementation of your custom model object.
- implement the `embed_texts()` method with **one and only one** parameter of type `List[str]` as the list of strings text to be embedded, and return a list of vectors of type `List[List[float]]`.
- implement the asynchronous `a_embed_text()` and `a_embed_texts()` method, with the same function signature as their respective synchronous versions. Since this is an asynchronous method, remember to use `async/await`.

:::note
If an asynchronous version of your embedding model does not exist, simply reuse the synchronous implementation:
"""
logger.info("When creating a custom embedding model, you should **ALWAYS**:")

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    ...
    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

"""
:::

Lastly, provide the custom embedding model through the `embedder` parameter  in the [`ContextConstructionConfig`](/docs/synthesizer-generate-from-docs#customize-context-construction) when calling any of the synthesis function:
"""
logger.info("Lastly, provide the custom embedding model through the `embedder` parameter  in the [`ContextConstructionConfig`](/docs/synthesizer-generate-from-docs#customize-context-construction) when calling any of the synthesis function:")

...

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    context_construction_config=ContextConstructionConfig(
        embedder=CustomEmbeddingModel()
    )
)

"""
:::tip
If you run into **invalid JSON errors** using custom models, you may want to consult [this guide](/guides/guides-using-custom-llms) on using custom LLMs for evaluation, as synthetic data generation also supports pydantic confinement for custom models.
:::
"""
logger.info("If you run into **invalid JSON errors** using custom models, you may want to consult [this guide](/guides/guides-using-custom-llms) on using custom LLMs for evaluation, as synthetic data generation also supports pydantic confinement for custom models.")

logger.info("\n\n[DONE]", bright=True)