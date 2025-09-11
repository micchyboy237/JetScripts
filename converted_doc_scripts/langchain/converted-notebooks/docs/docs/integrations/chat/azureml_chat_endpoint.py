from jet.logger import logger
from langchain_community.chat_models.azureml_endpoint import (
AzureMLEndpointApiType,
CustomOllamaChatContentFormatter,
)
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_core.messages import HumanMessage
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
sidebar_label: Azure ML Endpoint
---

# AzureMLChatOnlineEndpoint

>[Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning/) is a platform used to build, train, and deploy machine learning models. Users can explore the types of models to deploy in the Model Catalog, which provides foundational and general purpose models from different providers.
>
>In general, you need to deploy models in order to consume its predictions (inference). In `Azure Machine Learning`, [Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints) are used to deploy these models with a real-time serving. They are based on the ideas of `Endpoints` and `Deployments` which allow you to decouple the interface of your production workload from the implementation that serves it.

This notebook goes over how to use a chat model hosted on an `Azure Machine Learning Endpoint`.
"""
logger.info("# AzureMLChatOnlineEndpoint")


"""
## Set up

You must [deploy a model on Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-foundation-models?view=azureml-api-2#deploying-foundation-models-to-endpoints-for-inferencing) or [to Azure AI studio](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-open) and obtain the following parameters:

* `endpoint_url`: The REST endpoint url provided by the endpoint.
* `endpoint_api_type`: Use `endpoint_type='dedicated'` when deploying models to **Dedicated endpoints** (hosted managed infrastructure). Use `endpoint_type='serverless'` when deploying models using the **Pay-as-you-go** offering (model as a service).
* `endpoint_api_key`: The API key provided by the endpoint

## Content Formatter

The `content_formatter` parameter is a handler class for transforming the request and response of an AzureML endpoint to match with required schema. Since there are a wide range of models in the model catalog, each of which may process data differently from one another, a `ContentFormatterBase` class is provided to allow users to transform data to their liking. The following content formatters are provided:

* `CustomOllamaChatContentFormatter`: Formats request and response data for models like LLaMa2-chat that follow the Ollama API spec for request and response.

*Note: `langchain.chat_models.azureml_endpoint.LlamaChatContentFormatter` is being deprecated and replaced with `langchain.chat_models.azureml_endpoint.CustomOllamaChatContentFormatter`.*

You can implement custom content formatters specific for your model deriving from the class `langchain_community.llms.azureml_endpoint.ContentFormatterBase`.

## Examples

The following section contains examples about how to use this class:

### Example: Chat completions with real-time endpoints
"""
logger.info("## Set up")


chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
    endpoint_api_type=AzureMLEndpointApiType.dedicated,
    endpoint_content_formatter=CustomOllamaChatContentFormatter(),
)
response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")]
)
response

"""
### Example: Chat completions with pay-as-you-go deployments (model as a service)
"""
logger.info("### Example: Chat completions with pay-as-you-go deployments (model as a service)")

chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_content_formatter=CustomOllamaChatContentFormatter,
)
response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")]
)
response

"""
If you need to pass additional parameters to the model, use `model_kwargs` argument:
"""
logger.info("If you need to pass additional parameters to the model, use `model_kwargs` argument:")

chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_content_formatter=CustomOllamaChatContentFormatter,
    model_kwargs={"temperature": 0.8},
)

"""
Parameters can also be passed during invocation:
"""
logger.info("Parameters can also be passed during invocation:")

response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")],
    max_tokens=512,
)
response

logger.info("\n\n[DONE]", bright=True)