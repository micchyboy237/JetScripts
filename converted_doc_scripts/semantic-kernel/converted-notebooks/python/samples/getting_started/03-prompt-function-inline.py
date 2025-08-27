import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
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
# Running Prompt Functions Inline

Import Semantic Kernel SDK from pypi.org
"""
logger.info("# Running Prompt Functions Inline")

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

Add your [Azure Ollama Service key](https://learn.microsoft.com/azure/cognitive-services/openai/quickstart?pivots=programming-language-studio) settings to the `.env` file in the same folder:

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

The [previous notebook](./02-running-prompts-from-file.ipynb)
showed how to define a semantic function using a prompt template stored on a file.

In this notebook, we'll show how to use the Semantic Kernel to define functions inline with your python code. This can be useful in a few scenarios:

- Dynamically generating the prompt using complex rules at runtime
- Writing prompts by editing Python code instead of TXT files.
- Easily creating demos, like this document

Prompt templates are defined using the SK template language, which allows to reference variables and functions. Read [this doc](https://aka.ms/sk/howto/configurefunction) to learn more about the design decisions for prompt templating.

For now we'll use only the `{{$input}}` variable, and see more complex templates later.

Almost all semantic function prompts have a reference to `{{$input}}`, which is the default way
a user can import content from the context variables.

Prepare a semantic kernel instance first, loading also the AI service settings defined in the [Setup notebook](00-getting-started.ipynb):

Let's define our kernel for this example.
"""
logger.info("### Configuring the Kernel")


kernel = Kernel()



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
Let's use a prompt to create a semantic function used to summarize content, allowing for some creativity and a sufficient number of tokens.

The function will take in input the text to summarize.
"""
logger.info("Let's use a prompt to create a semantic function used to summarize content, allowing for some creativity and a sufficient number of tokens.")


prompt = """{{$input}}
Summarize the content above.
"""

if selectedService == Service.Ollama:
    execution_settings = OllamaChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7,
    )
elif selectedService == Service.AzureOllama:
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-35-turbo",
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="summarize",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

summarize = kernel.add_function(
    function_name="summarizeFunc",
    plugin_name="summarizePlugin",
    prompt_template_config=prompt_template_config,
)

"""
Set up some content to summarize, here's an extract about Demo, an ancient Greek poet, taken from Wikipedia (https://en.wikipedia.org/wiki/Demo_(ancient_Greek_poet)).
"""
logger.info("Set up some content to summarize, here's an extract about Demo, an ancient Greek poet, taken from Wikipedia (https://en.wikipedia.org/wiki/Demo_(ancient_Greek_poet)).")

input_text = """
Demo (ancient Greek poet)
From Wikipedia, the free encyclopedia
Demo or Damo (Greek: Δεμώ, Δαμώ; fl. c. AD 200) was a Greek woman of the Roman period, known for a single epigram, engraved upon the Colossus of Memnon, which bears her name. She speaks of herself therein as a lyric poetess dedicated to the Muses, but nothing is known of her life.[1]
Identity
Demo was evidently Greek, as her name, a traditional epithet of Demeter, signifies. The name was relatively common in the Hellenistic world, in Egypt and elsewhere, and she cannot be further identified. The date of her visit to the Colossus of Memnon cannot be established with certainty, but internal evidence on the left leg suggests her poem was inscribed there at some point in or after AD 196.[2]
Epigram
There are a number of graffiti inscriptions on the Colossus of Memnon. Following three epigrams by Julia Balbilla, a fourth epigram, in elegiac couplets, entitled and presumably authored by "Demo" or "Damo" (the Greek inscription is difficult to read), is a dedication to the Muses.[2] The poem is traditionally published with the works of Balbilla, though the internal evidence suggests a different author.[1]
In the poem, Demo explains that Memnon has shown her special respect. In return, Demo offers the gift for poetry, as a gift to the hero. At the end of this epigram, she addresses Memnon, highlighting his divine status by recalling his strength and holiness.[2]
Demo, like Julia Balbilla, writes in the artificial and poetic Aeolic dialect. The language indicates she was knowledgeable in Homeric poetry—'bearing a pleasant gift', for example, alludes to the use of that phrase throughout the Iliad and Odyssey.[a][2]
"""

"""
...and run the summary function:
"""

async def run_async_code_ed573181():
    async def run_async_code_009cd0cc():
        summary = await kernel.invoke(summarize, input=input_text)
        return summary
    summary = asyncio.run(run_async_code_009cd0cc())
    logger.success(format_json(summary))
    return summary
summary = asyncio.run(run_async_code_ed573181())
logger.success(format_json(summary))

logger.debug(summary)

"""
# Using ChatCompletion for Semantic Plugins

You can also use chat completion models (like `gpt-35-turbo` and `gpt4`) for creating plugins. Normally you would have to tweak the API to accommodate for a system and user role, but SK abstracts that away for you by using `kernel.add_service` and `AzureChatCompletion` or `OllamaChatCompletion`

Here's one more example of how to write an inline Semantic Function that gives a TLDR for a piece of text using a ChatCompletion model
"""
logger.info("# Using ChatCompletion for Semantic Plugins")

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


prompt = """
{{$input}}

Give me the TLDR in 5 words or less.
"""

text = """
    1) A robot may not injure a human being or, through inaction,
    allow a human being to come to harm.

    2) A robot must obey orders given it by human beings except where
    such orders would conflict with the First Law.

    3) A robot must protect its own existence as long as such protection
    does not conflict with the First or Second Law.
"""

if selectedService == Service.Ollama:
    execution_settings = OllamaChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7,
    )
elif selectedService == Service.AzureOllama:
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-35-turbo",
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

tldr_function = kernel.add_function(
    function_name="tldrFunction",
    plugin_name="tldrPlugin",
    prompt_template_config=prompt_template_config,
)

async def run_async_code_319f0627():
    summary = await kernel.invoke(tldr_function, input=text)
    return summary
summary = asyncio.run(run_async_code_319f0627())
logger.success(format_json(summary))

logger.debug(f"Output: {summary}")

logger.info("\n\n[DONE]", bright=True)