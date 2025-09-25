from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AzureOpenAIModel
from jet.logger import logger
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
# id: azure-ollama
title: Azure Ollama
sidebar_label: Azure Ollama
---

DeepEval allows you to directly integrate Azure Ollama models into all available LLM-based metrics. You can easily configure the model through the command line or directly within your python code.

### Command Line

Run the following command in your terminal to configure your deepeval environment to use Azure Ollama for all metrics.
"""
logger.info("# id: azure-ollama")

deepeval set-azure-ollama \
    --ollama-endpoint=<endpoint> \ # e.g. https://example-resource.azure.ollama.com/
    --ollama-api-key=<api_key> \
    --ollama-model-name=<model_name> \ # e.g. gpt-4.1
    --deployment-name=<deployment_name> \  # e.g. Test Deployment
    --ollama-api-version=<openai_api_version> \ # e.g. 2025-01-01-preview

"""
:::info
The CLI command above sets Azure Ollama as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Azure Ollama:
"""
logger.info("The CLI command above sets Azure Ollama as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Azure Ollama:")

deepeval unset-azure-ollama

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

Alternatively, you can specify your model directly in code using `AzureOpenAIModel` from DeepEval's model collection.

:::tip
This approach is ideal when you need to use separate models for specific metrics.
:::
"""
logger.info("### Python")


model = AzureOpenAIModel(
    model_name="gpt-4.1",
    deployment_name="Test Deployment",
    azure_openai_openai_api_version="2025-01-01-preview",
    azure_endpoint="https://example-resource.azure.ollama.com/",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **FIVE** mandatory and **ONE** optional parameters when creating an `AzureOpenAIModel`:

- `model_name`: A string specifying the name of the Azure Ollama model to use.
- `deployment_name`: A string specifying the name of your Azure Ollama deployment.
- `azure_openai_api_key`: A string specifying your Azure Ollama API key.
- `openai_api_version`: A string specifying the Ollama API version used in your deployment.
- `azure_endpoint`: A string specifying your Azure Ollama endpoint URL.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://learn.microsoft.com/en-us/azure/ai-foundry/ollama/reference#request-body).
:::

### Available Azure Ollama Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Azure Ollama's official documentation.
:::

Below is a list of commonly used Azure Ollama models:

- `gpt-4.1`
- `gpt-4.5-preview`
- `gpt-4o`
- `llama3.2`
- `gpt-4`
- `gpt-4-32k`
- `gpt-35-turbo`
- `gpt-35-turbo-16k`
- `gpt-35-turbo-instruct`
- `o1`
- `o1-mini`
- `o1-preview`
- `o3-mini`
"""
logger.info("### Available Azure Ollama Models")

logger.info("\n\n[DONE]", bright=True)