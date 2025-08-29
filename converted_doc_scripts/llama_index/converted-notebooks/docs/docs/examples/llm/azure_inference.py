from azure.identity import DefaultAzureCredential
from azure.identity.aio import (
DefaultAzureCredential as DefaultAzureCredentialAsync,
)
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.azure_inference import AzureAICompletionsModel
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/azure_openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Azure AI model inference

This notebook explains how to use `llama-index-llm-azure-inference` package with models deployed with the Azure AI model inference API in Azure AI studio or Azure Machine Learning. The package also support GitHub Models (Preview) endpoints.
"""
logger.info("# Azure AI model inference")

# %pip install llama-index-llms-azure-inference

"""
If you're opening this notebook on Google Colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this notebook on Google Colab, you will probably need to install LlamaIndex 🦙.")

# %pip install llama-index

"""
## Prerequisites

The Azure AI model inference is an API that allows developers to get access to a variety of models hosted on Azure AI using a consistent schema. You can use `llama-index-llms-azure-inference` integration package with models that support this API, including models deployed to Azure AI serverless API endpoints and a subset of models from Managed Inference. To read more about the API specification and the models that support it see [Azure AI model inference API](https://aka.ms/azureai/modelinference).

To run this tutorial you need:

1. Create an [Azure subscription](https://azure.microsoft.com).
2. Create an Azure AI hub resource as explained at [How to create and manage an Azure AI Studio hub](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/create-azure-ai-resource).
3. Deploy one model supporting the [Azure AI model inference API](https://aka.ms/azureai/modelinference). In this example we use a `Mistral-Large` deployment. 

    * You can follow the instructions at [Deploy models as serverless APIs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-serverless).

Alternatively, you can use GitHub Models endpoints with this integration, including the free tier experience. Read more about [GitHub models](https://github.com/marketplace/models).

## Environment Setup

Follow this steps to get the information you need from the model you want to use:

1. Go to the [Azure AI Foundry (formerly Azure AI Studio)](https://ai.azure.com/) or [Azure Machine Learning studio](https://ml.azure.com), depending on the product you are using.
2. Go to deployments (endpoints in Azure Machine Learning) and select the model you have deployed as indicated in the prerequisites.
3. Copy the endpoint URL and the key.
    
> If your model was deployed with Microsoft Entra ID support, you don't need a key.

In this scenario, we have placed both the endpoint URL and key in the following environment variables:
"""
logger.info("## Prerequisites")


os.environ["AZURE_INFERENCE_ENDPOINT"] = "<your-endpoint>"
os.environ["AZURE_INFERENCE_CREDENTIAL"] = "<your-credential>"

"""
## Connect to your deployment and endpoint

To use LLMs deployed in Azure AI studio or Azure Machine Learning you need the endpoint and credentials to connect to it. The parameter `model_name` is not required for endpoints serving a single model, like Managed Online Endpoints.
"""
logger.info("## Connect to your deployment and endpoint")


llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
)

"""
Alternatively, if you endpoint support Microsoft Entra ID, you can use the following code to create the client:
"""
logger.info("Alternatively, if you endpoint support Microsoft Entra ID, you can use the following code to create the client:")


llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

"""
> Note: When using Microsoft Entra ID, make sure that the endpoint was deployed with that authentication method and that you have the required permissions to invoke it.

If you are planning to use asynchronous calling, it's a best practice to use the asynchronous version for the credentials:
"""
logger.info("If you are planning to use asynchronous calling, it's a best practice to use the asynchronous version for the credentials:")


llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=DefaultAzureCredentialAsync(),
)

"""
If your endpoint is serving more than one model, like [GitHub Models](https://github.com/marketplace/models) or Azure AI Services, then you have to indicate the parameter `model_name`:
"""
logger.info("If your endpoint is serving more than one model, like [GitHub Models](https://github.com/marketplace/models) or Azure AI Services, then you have to indicate the parameter `model_name`:")

llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    model_name="mistral-large",  # change it to the model you want to use
)

"""
## Use the model

Use the `complete` endpoint for text completion. Ihe `complete` method is still available for model of type `chat-completions`. On those cases, your input text is converted to a message with `role="user"`.
"""
logger.info("## Use the model")

response = llm.complete("The sky is a beautiful blue and")
logger.debug(response)

response = llm.stream_complete("The sky is a beautiful blue and")
for r in response:
    logger.debug(r.delta, end="")

"""
Use the `chat` endpoint for conversation
"""
logger.info("Use the `chat` endpoint for conversation")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with colorful personality."
    ),
    ChatMessage(role="user", content="Hello"),
]

response = llm.chat(messages)
logger.debug(response)

response = llm.stream_chat(messages)
for r in response:
    logger.debug(r.delta, end="")

"""
Rather than adding same parameters to each chat or completion call, you can set them at the client instance.
"""
logger.info("Rather than adding same parameters to each chat or completion call, you can set them at the client instance.")

llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    temperature=0.0,
    model_kwargs={"top_p": 1.0},
)

response = llm.complete("The sky is a beautiful blue and")
logger.debug(response)

"""
For parameters extra parameters that are not supported by the Azure AI model inference API but that are available in the underlying model, you can use the `model_extras` argument. In the following example, the parameter `safe_prompt`, only available for Mistral models, is being passed.
"""
logger.info("For parameters extra parameters that are not supported by the Azure AI model inference API but that are available in the underlying model, you can use the `model_extras` argument. In the following example, the parameter `safe_prompt`, only available for Mistral models, is being passed.")

llm = AzureAICompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    temperature=0.0,
    model_kwargs={"model_extras": {"safe_prompt": True}},
)

response = llm.complete("The sky is a beautiful blue and")
logger.debug(response)

"""
## Additional resources

To learn more about this integration visit [Getting starting with LlamaIndex and Azure AI](https://aka.ms/azureai/llamaindex).
"""
logger.info("## Additional resources")

logger.info("\n\n[DONE]", bright=True)