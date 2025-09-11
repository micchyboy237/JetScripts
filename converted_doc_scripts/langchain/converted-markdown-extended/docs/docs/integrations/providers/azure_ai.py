from jet.logger import logger
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
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
# Azure AI

All functionality related to [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/developer/python/get-started) and its related projects.

Integration packages for Azure AI, Dynamic Sessions, SQL Server are maintained in
the [langchain-azure](https://github.com/langchain-ai/langchain-azure) repository.

## Chat models

We recommend developers start with the (`langchain-azure-ai`) to access all the models available in [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview).

### Azure AI Chat Completions Model

Access models like Azure Ollama, DeepSeek R1, Cohere, Phi and Mistral using the `AzureAIChatCompletionsModel` class.
"""
logger.info("# Azure AI")

pip install -U langchain-azure-ai

"""
Configure your API key and Endpoint.
"""
logger.info("Configure your API key and Endpoint.")

export AZURE_INFERENCE_CREDENTIAL=your-api-key
export AZURE_INFERENCE_ENDPOINT=your-endpoint

"""

"""


llm = AzureAIChatCompletionsModel(
    model_name="gpt-4o",
    api_version="2024-05-01-preview",
)

llm.invoke('Tell me a joke and include some emojis')

"""
## Embedding models

### Azure AI model inference for embeddings
"""
logger.info("## Embedding models")

pip install -U langchain-azure-ai

"""
Configure your API key and Endpoint.
"""
logger.info("Configure your API key and Endpoint.")

export AZURE_INFERENCE_CREDENTIAL=your-api-key
export AZURE_INFERENCE_ENDPOINT=your-endpoint

"""

"""


embed_model = AzureAIEmbeddingsModel(
    model_name="text-embedding-ada-002"
)

logger.info("\n\n[DONE]", bright=True)