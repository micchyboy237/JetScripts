import asyncio
from jet.transformers.formatters import format_json
from IPython.display import clear_output
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.hugging_face import (  # noqa: F401
HuggingFacePromptExecutionSettings,
HuggingFaceTextCompletion,
)
from semantic_kernel.connectors.ai.open_ai import (
AzureChatCompletion,
AzureChatPromptExecutionSettings,  # noqa: F401
AzureTextCompletion,
OllamaChatCompletion,
OllamaChatPromptExecutionSettings,  # noqa: F401
OllamaTextCompletion,
OllamaTextPromptExecutionSettings,  # noqa: F401
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.contents import ChatHistory
from services import Service
import os
import shutil
import sys
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Multiple Results

In this notebook we show how you can in a single request, have the LLM model return multiple results per prompt. This is useful for running experiments where you want to evaluate the robustness of your prompt and the parameters of your config against a particular large language model.

Import Semantic Kernel SDK from pypi.org
"""
logger.info("# Multiple Results")

# %pip install -U semantic-kernel

__version__

"""
Initial configuration for the notebook to run properly.
"""
logger.info("Initial configuration for the notebook to run properly.")


notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)


sys.path.append(grandparent_dir)

"""
### Configuring the Kernel

Let's get started with the necessary configuration to run Semantic Kernel. For Notebooks, we require a `.env` file with the proper settings for the model you use. Create a new file named `.env` and place it in this directory. Copy the contents of the `.env.example` file from this directory and paste it into the `.env` file that you just created.

**NOTE: Please make sure to include `GLOBAL_LLM_SERVICE` set to either Ollama, AzureOllama, or HuggingFace in your .env file. If this setting is not included, the Service will default to AzureOllama.**

#### Option 1: using Ollama

Add your [Ollama Key](https://openai.com/product/) key to your `.env` file (org Id only if you have multiple orgs):

```
GLOBAL_LLM_SERVICE="Ollama"
# OPENAI_API_KEY="sk-..."
OPENAI_ORG_ID=""
OPENAI_CHAT_MODEL_ID=""
OPENAI_TEXT_MODEL_ID=""
OPENAI_EMBEDDING_MODEL_ID=""
```
The names should match the names used in the `.env` file, as shown above.

#### Option 2: using Azure Ollama

Add your [Azure Open AI Service key](https://learn.microsoft.com/azure/cognitive-services/openai/quickstart?pivots=programming-language-studio) settings to the `.env` file in the same folder:

```
GLOBAL_LLM_SERVICE="AzureOllama"
# AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://..."
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="..."
AZURE_OPENAI_TEXT_DEPLOYMENT_NAME="..."
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="..."
AZURE_OPENAI_API_VERSION="..."
```
The names should match the names used in the `.env` file, as shown above.

# As alternative to `AZURE_OPENAI_API_KEY`, it's possible to authenticate using `credential` parameter, more information here: [Azure Identity](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme).

In the following example, `AzureCliCredential` is used. To authenticate using Azure CLI:

1. Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
2. Run `az login` command in terminal and follow the authentication steps.

For more advanced configuration, please follow the steps outlined in the [setup guide](./CONFIGURING_THE_KERNEL.md).

We will load our settings and get the LLM service to use for the notebook.
"""
logger.info("### Configuring the Kernel")



service_settings = ServiceSettings()

selectedService = (
    Service.AzureOllama
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
logger.debug(f"Using service type: {selectedService}")

"""
First, we will set up the text and chat services we will be submitting prompts to.
"""
logger.info("First, we will set up the text and chat services we will be submitting prompts to.")


kernel = Kernel()

service_id = None
if selectedService == Service.Ollama:

    service_id = "default"
    oai_chat_service = OllamaChatCompletion(
        service_id="oai_chat",
    )
    oai_text_service = OllamaTextCompletion(
        service_id="oai_text",
    )
elif selectedService == Service.AzureOllama:


    credential = AzureCliCredential()
    service_id = "default"
    aoai_chat_service = AzureChatCompletion(service_id="aoai_chat", credential=credential)
    aoai_text_service = AzureTextCompletion(service_id="aoai_text", credential=credential)

if selectedService == Service.HuggingFace:

    hf_text_service = HuggingFaceTextCompletion(service_id="hf_text", ai_model_id="distilgpt2", task="text-generation")

"""
Next, we'll set up the completion request settings for text completion services.
"""
logger.info("Next, we'll set up the completion request settings for text completion services.")

oai_text_prompt_execution_settings = OllamaTextPromptExecutionSettings(
    service="oai_text",
    extension_data={
        "max_tokens": 80,
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "number_of_responses": 3,
    },
)

"""
## Multiple Open AI Text Completions
"""
logger.info("## Multiple Open AI Text Completions")

if selectedService == Service.Ollama:
    prompt = "What is the purpose of a rubber duck?"

    async def run_async_code_810b9aed():
        async def run_async_code_5c5f58d4():
            results = await oai_text_service.get_text_contents(prompt=prompt, settings=oai_text_prompt_execution_settings)
            return results
        results = asyncio.run(run_async_code_5c5f58d4())
        logger.success(format_json(results))
        return results
    results = asyncio.run(run_async_code_810b9aed())
    logger.success(format_json(results))

    for i, result in enumerate(results):
        logger.debug(f"Result {i + 1}: {result}")

"""
## Multiple Azure Open AI Text Completions
"""
logger.info("## Multiple Azure Open AI Text Completions")

if selectedService == Service.AzureOllama:
    prompt = "provide me a list of possible meanings for the acronym 'ORLD'"

    async def run_async_code_888c0176():
        async def run_async_code_1d68336e():
            results = await aoai_text_service.get_text_contents(prompt=prompt, settings=oai_text_prompt_execution_settings)
            return results
        results = asyncio.run(run_async_code_1d68336e())
        logger.success(format_json(results))
        return results
    results = asyncio.run(run_async_code_888c0176())
    logger.success(format_json(results))

    for i, result in enumerate(results):
        logger.debug(f"Result {i + 1}: {result}")

"""
## Multiple Hugging Face Text Completions
"""
logger.info("## Multiple Hugging Face Text Completions")

if selectedService == Service.HuggingFace:
    hf_prompt_execution_settings = HuggingFacePromptExecutionSettings(
        service_id="hf_text",
        extension_data={"max_new_tokens": 80, "temperature": 0.7, "top_p": 1, "num_return_sequences": 3},
    )

if selectedService == Service.HuggingFace:
    prompt = "The purpose of a rubber duck is"

    async def run_async_code_605af86d():
        async def run_async_code_a059b9c7():
            results = await hf_text_service.get_text_contents(prompt=prompt, settings=hf_prompt_execution_settings)
            return results
        results = asyncio.run(run_async_code_a059b9c7())
        logger.success(format_json(results))
        return results
    results = asyncio.run(run_async_code_605af86d())
    logger.success(format_json(results))

    for i, result in enumerate(results):
        logger.debug(f"Result {i + 1}: {result}")

"""
Here, we're setting up the settings for Chat completions.
"""
logger.info("Here, we're setting up the settings for Chat completions.")

oai_chat_prompt_execution_settings = OllamaChatPromptExecutionSettings(
    service_id="oai_chat",
    max_tokens=80,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    number_of_responses=3,
)

"""
## Multiple Ollama Chat Completions
"""
logger.info("## Multiple Ollama Chat Completions")


if selectedService == Service.Ollama:
    chat = ChatHistory()
    chat.add_user_message(
        "It's a beautiful day outside, birds are singing, flowers are blooming. On days like these, kids like you..."
    )
    async def async_func_7():
        results = await oai_chat_service.get_chat_message_contents(
            chat_history=chat, settings=oai_chat_prompt_execution_settings
        )
        return results
    results = asyncio.run(async_func_7())
    logger.success(format_json(results))

    for i, result in enumerate(results):
        logger.debug(f"Result {i + 1}: {result!s}")

"""
## Multiple Azure Ollama Chat Completions
"""
logger.info("## Multiple Azure Ollama Chat Completions")

az_oai_prompt_execution_settings = AzureChatPromptExecutionSettings(
    service_id="aoai_chat",
    max_tokens=80,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    number_of_responses=3,
)

if selectedService == Service.AzureOllama:
    content = (
        "Tomorrow is going to be a great day, I can feel it. I'm going to wake up early, go for a run, and then..."
    )
    chat = ChatHistory()
    chat.add_user_message(content)
    async def async_func_16():
        results = await aoai_chat_service.get_chat_message_contents(
            chat_history=chat, settings=az_oai_prompt_execution_settings
        )
        return results
    results = asyncio.run(async_func_16())
    logger.success(format_json(results))

    for i, result in enumerate(results):
        logger.debug(f"Result {i + 1}: {result!s}")

"""
## Streaming Multiple Results

Here is an example pattern if you want to stream your multiple results. Note that this is not supported for Hugging Face text completions at this time.
"""
logger.info("## Streaming Multiple Results")

if selectedService == Service.Ollama:


    clear_command = "cls" if os.name == "nt" else "clear"

    chat = ChatHistory()
    chat.add_user_message("what is the purpose of a rubber duck?")

    stream = oai_chat_service.get_streaming_chat_message_contents(
        chat_history=chat, settings=oai_chat_prompt_execution_settings
    )
    number_of_responses = oai_chat_prompt_execution_settings.number_of_responses
    texts = [""] * number_of_responses

    last_clear_time = time.time()
    clear_interval = 0.5  # seconds


    async for results in stream:
        current_time = time.time()

        for result in results:
            texts[result.choice_index] += str(result)

        if current_time - last_clear_time > clear_interval:
            clear_output(wait=True)
            for idx, text in enumerate(texts):
                logger.debug(f"Result {idx + 1}: {text}")
            last_clear_time = current_time

    logger.debug("----------------------------------------")

logger.info("\n\n[DONE]", bright=True)