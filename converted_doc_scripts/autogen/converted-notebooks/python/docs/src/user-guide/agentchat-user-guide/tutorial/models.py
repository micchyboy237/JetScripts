import asyncio
from jet.transformers.formatters import format_json
from autogen_core import EVENT_LOGGER_NAME
from autogen_core import Image
from autogen_core.models import ModelInfo
from autogen_core.models import UserMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.azure import AzureAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import AzureOllamaChatCompletionClient
from autogen_ext.models.openai import OllamaChatCompletionClient
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from pathlib import Path
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion, AnthropicChatPromptExecutionSettings
from semantic_kernel.memory.null_memory import NullMemory
import logging
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Models

In many cases, agents need access to LLM model services such as Ollama, Azure Ollama, or local models. Since there are many different providers with different APIs, `autogen-core` implements a protocol for model clients and `autogen-ext` implements a set of model clients for popular model services. AgentChat can use these model clients to interact with model services. 

This section provides a quick overview of available model clients.
For more details on how to use them directly, please refer to [Model Clients](../../core-user-guide/components/model-clients.ipynb) in the Core API documentation.

```{note}
See {py:class}`~autogen_ext.models.cache.ChatCompletionCache` for a caching wrapper to use with the following clients.
```

## Log Model Calls

AutoGen uses standard Python logging module to log events like model calls and responses.
The logger name is {py:attr}`autogen_core.EVENT_LOGGER_NAME`, and the event type is `LLMCall`.
"""
logger.info("# Models")



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

"""
## Ollama

To access Ollama models, install the `openai` extension, which allows you to use the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`.
"""
logger.info("## Ollama")

pip install "autogen-ext[openai]"

"""
You will also need to obtain an [API key](https://platform.openai.com/account/api-keys) from Ollama.
"""
logger.info("You will also need to obtain an [API key](https://platform.openai.com/account/api-keys) from Ollama.")


openai_model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
)

"""
To test the model client, you can use the following code:
"""
logger.info("To test the model client, you can use the following code:")


async def run_async_code_6c382726():
    async def run_async_code_e750dad5():
        result = await openai_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_e750dad5())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_6c382726())
logger.success(format_json(result))
logger.debug(result)
async def run_async_code_72fad18f():
    await openai_model_client.close()
    return 
 = asyncio.run(run_async_code_72fad18f())
logger.success(format_json())

"""
```{note}
You can use this client with models hosted on Ollama-compatible endpoints, however, we have not tested this functionality.
See {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient` for more information.
```

## Azure Ollama

Similarly, install the `azure` and `openai` extensions to use the {py:class}`~autogen_ext.models.openai.AzureOllamaChatCompletionClient`.
"""
logger.info("## Azure Ollama")

pip install "autogen-ext[openai,azure]"

"""
To use the client, you need to provide your deployment id, Azure Cognitive Services endpoint, api version, and model capabilities.
For authentication, you can either provide an API key or an Azure Active Directory (AAD) token credential.

The following code snippet shows how to use AAD authentication.
The identity used must be assigned the [Cognitive Services Ollama User](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/role-based-access-control#cognitive-services-openai-user) role.
"""
logger.info("To use the client, you need to provide your deployment id, Azure Cognitive Services endpoint, api version, and model capabilities.")


token_provider = AzureTokenProvider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

az_model_client = AzureOllamaChatCompletionClient(
    azure_deployment="{your-azure-deployment}",
    model="{model-name, such as gpt-4o}",
    api_version="2024-06-01",
    azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
    azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
)

async def run_async_code_759f2e10():
    async def run_async_code_383d527f():
        result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_383d527f())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_759f2e10())
logger.success(format_json(result))
logger.debug(result)
async def run_async_code_8aa14161():
    await az_model_client.close()
    return 
 = asyncio.run(run_async_code_8aa14161())
logger.success(format_json())

"""
See [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity#chat-completions) for how to use the Azure client directly or for more information.

## Azure AI Foundry

[Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/) (previously known as Azure AI Studio) offers models hosted on Azure.
To use those models, you use the {py:class}`~autogen_ext.models.azure.AzureAIChatCompletionClient`.

You need to install the `azure` extra to use this client.
"""
logger.info("## Azure AI Foundry")

pip install "autogen-ext[azure]"

"""
Below is an example of using this client with the Phi-4 model from [GitHub Marketplace](https://github.com/marketplace/models).
"""
logger.info("Below is an example of using this client with the Phi-4 model from [GitHub Marketplace](https://github.com/marketplace/models).")



client = AzureAIChatCompletionClient(
    model="Phi-4",
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    model_info={
        "json_output": False,
        "function_calling": False,
        "vision": False,
        "family": "unknown",
        "structured_output": False,
    },
)

async def run_async_code_2defc511():
    async def run_async_code_eab808f6():
        result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_eab808f6())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_2defc511())
logger.success(format_json(result))
logger.debug(result)
async def run_async_code_c4f3b18a():
    await client.close()
    return 
 = asyncio.run(run_async_code_c4f3b18a())
logger.success(format_json())

"""
## Anthropic (experimental)

To use the {py:class}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient`, you need to install the `anthropic` extra. Underneath, it uses the `anthropic` python sdk to access the models.
You will also need to obtain an [API key](https://console.anthropic.com) from Anthropic.
"""
logger.info("## Anthropic (experimental)")




anthropic_client = AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219")
async def run_async_code_f147f07d():
    async def run_async_code_66cce63a():
        result = await anthropic_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_66cce63a())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_f147f07d())
logger.success(format_json(result))
logger.debug(result)
async def run_async_code_d246b63f():
    await anthropic_client.close()
    return 
 = asyncio.run(run_async_code_d246b63f())
logger.success(format_json())

"""
## Ollama(experimental)

[Ollama](https://ollama.com/) is a local model server that can run models locally on your machine.

```{note}
Small local models are typically not as capable as larger models on the cloud.
For some tasks they may not perform as well and the output may be suprising.
```

To use Ollama, install the `ollama` extension and use the {py:class}`~autogen_ext.models.ollama.OllamaChatCompletionClient`.
"""
logger.info("## Ollama(experimental)")

pip install -U "autogen-ext[ollama]"


ollama_model_client = OllamaChatCompletionClient(model="llama3.2")

async def run_async_code_76d47701():
    async def run_async_code_99ca494a():
        response = await ollama_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return response
    response = asyncio.run(run_async_code_99ca494a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_76d47701())
logger.success(format_json(response))
logger.debug(response)
async def run_async_code_d4cf5a76():
    await ollama_model_client.close()
    return 
 = asyncio.run(run_async_code_d4cf5a76())
logger.success(format_json())

"""
## Gemini (experimental)

Gemini currently offers [an Ollama-compatible API (beta)](https://ai.google.dev/gemini-api/docs/openai).
So you can use the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient` with the Gemini API.

```{note}
While some model providers may offer Ollama-compatible APIs, they may still have minor differences.
For example, the `finish_reason` field may be different in the response.

```
"""
logger.info("## Gemini (experimental)")


model_client = OllamaChatCompletionClient(
    model="gemini-1.5-flash-8b",
)

async def run_async_code_536a0273():
    async def run_async_code_f25cb54e():
        response = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return response
    response = asyncio.run(run_async_code_f25cb54e())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_536a0273())
logger.success(format_json(response))
logger.debug(response)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
Also, as Gemini adds new models, you may need to define the models capabilities via the model_info field. For example, to use `gemini-2.0-flash-lite` or a similar new model, you can use the following code:

```python 

model_client = OllamaChatCompletionClient(
    model="gemini-2.0-flash-lite",
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True)
    # api_key="GEMINI_API_KEY",
)

response = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
logger.debug(response)
await model_client.close()
```

## Llama API (experimental)

[Llama API](https://llama.developer.meta.com?utm_source=partner-autogen&utm_medium=readme) is the Meta's first party API offering. It currently offers an [Ollama compatible endpoint](https://llama.developer.meta.com/docs/features/compatibility).
So you can use the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient` with the Llama API.

This endpoint fully supports the following Ollama client library features:
* Chat completions
* Model selection
* Temperature/sampling
* Streaming
* Image understanding
* Structured output (JSON mode)
* Function calling (tools)
"""
logger.info("# api_key="GEMINI_API_KEY",")



model_client = OllamaChatCompletionClient(
    model="Llama-4-Scout-17B-16E-Instruct-FP8",
)

async def run_async_code_20a5ddd0():
    async def run_async_code_e16b4331():
        response = await model_client.create([UserMessage(content="Write me a poem", source="user")])
        return response
    response = asyncio.run(run_async_code_e16b4331())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_20a5ddd0())
logger.success(format_json(response))
logger.debug(response)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

model_client = OllamaChatCompletionClient(
    model="Llama-4-Maverick-17B-128E-Instruct-FP8",
)
image = Image.from_file(Path("test.png"))

async def run_async_code_de33fdf0():
    async def run_async_code_fd56e21e():
        response = await model_client.create([UserMessage(content=["What is in this image", image], source="user")])
        return response
    response = asyncio.run(run_async_code_fd56e21e())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_de33fdf0())
logger.success(format_json(response))
logger.debug(response)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
## Semantic Kernel Adapter

The {py:class}`~autogen_ext.models.semantic_kernel.SKChatCompletionAdapter`
allows you to use Semantic kernel model clients as a
{py:class}`~autogen_core.models.ChatCompletionClient` by adapting them to the required interface.

You need to install the relevant provider extras to use this adapter. 

The list of extras that can be installed:

- `semantic-kernel-anthropic`: Install this extra to use Anthropic models.
- `semantic-kernel-google`: Install this extra to use Google Gemini models.
- `semantic-kernel-ollama`: Install this extra to use Ollama models.
- `semantic-kernel-mistralai`: Install this extra to use MistralAI models.
- `semantic-kernel-aws`: Install this extra to use AWS models.
- `semantic-kernel-hugging-face`: Install this extra to use Hugging Face models.

For example, to use Anthropic models, you need to install `semantic-kernel-anthropic`.
"""
logger.info("## Semantic Kernel Adapter")



"""
To use this adapter, you need create a Semantic Kernel model client and pass it to the adapter.

For example, to use the Anthropic model:
"""
logger.info("To use this adapter, you need create a Semantic Kernel model client and pass it to the adapter.")



sk_client = AnthropicChatCompletion(
    ai_model_id="claude-3-5-sonnet-20241022",
#     api_key=os.environ["ANTHROPIC_API_KEY"],
    service_id="my-service-id",  # Optional; for targeting specific services within Semantic Kernel
)
settings = AnthropicChatPromptExecutionSettings(
    temperature=0.2,
)

anthropic_model_client = SKChatCompletionAdapter(
    sk_client, kernel=Kernel(memory=NullMemory()), prompt_settings=settings
)

async def async_func_21():
    model_result = await anthropic_model_client.create(
        messages=[UserMessage(content="What is the capital of France?", source="User")]
    )
    return model_result
model_result = asyncio.run(async_func_21())
logger.success(format_json(model_result))
logger.debug(model_result)
async def run_async_code_7ddc9cc5():
    await anthropic_model_client.close()
    return 
 = asyncio.run(run_async_code_7ddc9cc5())
logger.success(format_json())

"""
Read more about the [Semantic Kernel Adapter](../../../reference/python/autogen_ext.models.semantic_kernel.rst).
"""
logger.info("Read more about the [Semantic Kernel Adapter](../../../reference/python/autogen_ext.models.semantic_kernel.rst).")

logger.info("\n\n[DONE]", bright=True)