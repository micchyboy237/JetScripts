import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from services import Service
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# How to run a prompt plugins from file

Now that you're familiar with Kernel basics, let's see how the kernel allows you to run Prompt Plugins and Prompt Functions stored on disk.

A Prompt Plugin is a collection of Semantic Functions, where each function is defined with natural language that can be provided with a text file.

Refer to our [glossary](https://github.com/microsoft/semantic-kernel/blob/main/docs/GLOSSARY.md) for an in-depth guide to the terms.

The repository includes some examples under the [samples](https://github.com/microsoft/semantic-kernel/tree/main/samples) folder.

For instance, [this](../../../prompt_template_samples/FunPlugin/Joke/skprompt.txt) is the **Joke function** part of the **FunPlugin plugin**:

Import Semantic Kernel SDK from pypi.org
"""
logger.info("# How to run a prompt plugins from file")

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

Let's move on to learning what prompts are and how to write them.

```
WRITE EXACTLY ONE JOKE or HUMOROUS STORY ABOUT THE TOPIC BELOW.
JOKE MUST BE:
- G RATED
- WORKPLACE/FAMILY SAFE
NO SEXISM, RACISM OR OTHER BIAS/BIGOTRY.
BE CREATIVE AND FUNNY. I WANT TO LAUGH.
+++++
{{$input}}
+++++
```

Note the special **`{{$input}}`** token, which is a variable that is automatically passed when invoking the function, commonly referred to as a "function parameter".

We'll explore later how functions can accept multiple variables, as well as invoke other functions.

In the same folder you'll notice a second [config.json](../../../prompt_template_samples/FunPlugin/Joke/config.json) file. The file is optional, and is used to set some parameters for large language models like Temperature, TopP, Stop Sequences, etc.

```
{
  "schema": 1,
  "description": "Generate a funny joke",
  "execution_settings": {
    "default": {
      "max_tokens": 1000,
      "temperature": 0.9,
      "top_p": 0.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0
    }
  },
  "input_variables": [
    {
      "name": "input",
      "description": "Joke subject",
      "default": ""
    },
    {
      "name": "style",
      "description": "Give a hint about the desired joke style",
      "default": ""
    }
  ]
}

```

Given a prompt function defined by these files, this is how to load and use a file based prompt function.

Load and configure the kernel, as usual, loading also the AI service settings defined in the [Setup notebook](00-getting-started.ipynb):
"""
logger.info("### Configuring the Kernel")


kernel = Kernel()

"""
We will load our settings and get the LLM service to use for the notebook.
"""
logger.info("We will load our settings and get the LLM service to use for the notebook.")



service_settings = ServiceSettings()

selectedService = (
    Service.AzureOllama
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
logger.debug(f"Using service type: {selectedService}")

"""
Let's load our settings and validate that the required ones exist.
"""
logger.info("Let's load our settings and validate that the required ones exist.")



service_settings = ServiceSettings()

selectedService = (
    Service.AzureOllama
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
logger.debug(f"Using service type: {selectedService}")

"""
We now configure our Chat Completion service on the kernel.
"""
logger.info("We now configure our Chat Completion service on the kernel.")

kernel.remove_all_services()

service_id = None
if selectedService == Service.Ollama:

    service_id = "default"
    kernel.add_service(
        OllamaChatCompletion(
            service_id=service_id,
        ),
    )
elif selectedService == Service.AzureOllama:


    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(service_id=service_id, credential=AzureCliCredential()),
    )

"""
Import the plugin and all its functions:
"""
logger.info("Import the plugin and all its functions:")

plugins_directory = "../../../prompt_template_samples/"

funFunctions = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="FunPlugin")

jokeFunction = funFunctions["Joke"]

"""
How to use the plugin functions, e.g. generate a joke about "_time travel to dinosaur age_":
"""
logger.info("How to use the plugin functions, e.g. generate a joke about "_time travel to dinosaur age_":")

async def run_async_code_62beadab():
    async def run_async_code_8aac13f2():
        result = await kernel.invoke(jokeFunction, input="travel to dinosaur age", style="silly")
        return result
    result = asyncio.run(run_async_code_8aac13f2())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_62beadab())
logger.success(format_json(result))
logger.debug(result)

"""
Great, now that you know how to load a plugin from disk, let's show how you can [create and run a prompt function inline.](./03-prompt-function-inline.ipynb)
"""
logger.info("Great, now that you know how to load a plugin from disk, let's show how you can [create and run a prompt function inline.](./03-prompt-function-inline.ipynb)")

logger.info("\n\n[DONE]", bright=True)