import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from services import Service
from typing import Annotated
import os
import random
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Running Native Functions

Two of the previous notebooks showed how to [execute semantic functions inline](./03-semantic-function-inline.ipynb) and how to [run prompts from a file](./02-running-prompts-from-file.ipynb).

In this notebook, we'll show how to use native functions from a file. We will also show how to call semantic functions from native functions.

This can be useful in a few scenarios:

- Writing logic around how to run a prompt that changes the prompt's outcome.
- Using external data sources to gather data to concatenate into your prompt.
- Validating user input data prior to sending it to the LLM prompt.

Native functions are defined using standard Python code. The structure is simple, but not well documented at this point.

The following examples are intended to help guide new users towards successful native & semantic function use with the SK Python framework.

Prepare a semantic kernel instance first, loading also the AI service settings defined in the [Setup notebook](00-getting-started.ipynb):

Import Semantic Kernel SDK from pypi.org
"""
logger.info("# Running Native Functions")

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
We now configure our Chat Completion service on the kernel.
"""
logger.info("We now configure our Chat Completion service on the kernel.")


kernel = Kernel()

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
Let's create a **native** function that gives us a random number between 3 and a user input as the upper limit. We'll use this number to create 3-x paragraphs of text when passed to a semantic function.

First, let's create our native function.
"""
logger.info("Let's create a **native** function that gives us a random number between 3 and a user input as the upper limit. We'll use this number to create 3-x paragraphs of text when passed to a semantic function.")




class GenerateNumberPlugin:
    """
    Description: Generate a number between 3-x.
    """

    @kernel_function(
        description="Generate a random number between 3-x",
        name="GenerateNumberThreeOrHigher",
    )
    def generate_number_three_or_higher(self, input: str) -> str:
        """
        Generate a number between 3-<input>
        Example:
            "8" => rand(3,8)
        Args:
            input -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(3, int(input)))
        except ValueError as e:
            logger.debug(f"Invalid input {input}")
            raise e

"""
Next, let's create a semantic function that accepts a number as `{{$input}}` and generates that number of paragraphs about two Corgis on an adventure. `$input` is a default variable semantic functions can use.
"""
logger.info("Next, let's create a semantic function that accepts a number as `{{$input}}` and generates that number of paragraphs about two Corgis on an adventure. `$input` is a default variable semantic functions can use.")


prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$input}} paragraphs long. It must be this length.
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
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStory",
    plugin_name="CorgiPlugin",
    prompt_template_config=prompt_template_config,
)

generate_number_plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")

generate_number_three_or_higher = generate_number_plugin["GenerateNumberThreeOrHigher"]
async def run_async_code_a28c46d8():
    async def run_async_code_67800c73():
        number_result = await generate_number_three_or_higher(kernel, input=6)
        return number_result
    number_result = asyncio.run(run_async_code_67800c73())
    logger.success(format_json(number_result))
    return number_result
number_result = asyncio.run(run_async_code_a28c46d8())
logger.success(format_json(number_result))
logger.debug(number_result)

async def run_async_code_c2ece15a():
    async def run_async_code_d6db4526():
        story = await corgi_story.invoke(kernel, input=number_result.value)
        return story
    story = asyncio.run(run_async_code_d6db4526())
    logger.success(format_json(story))
    return story
story = asyncio.run(run_async_code_c2ece15a())
logger.success(format_json(story))

"""
_Note: depending on which model you're using, it may not respond with the proper number of paragraphs._
"""

logger.debug(f"Generating a corgi story exactly {number_result.value} paragraphs long.")
logger.debug("=====================================================")
logger.debug(story)

"""
## Kernel Functions with Annotated Parameters

That works! But let's expand on our example to make it more generic.

For the native function, we'll introduce the lower limit variable. This means that a user will input two numbers and the number generator function will pick a number between the first and second input.

We'll make use of the Python's `Annotated` class to hold these variables.
"""
logger.info("## Kernel Functions with Annotated Parameters")

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
Let's start with the native function. Notice that we're add the `@kernel_function` decorator that holds the name of the function as well as an optional description. The input parameters are configured as part of the function's signature, and we use the `Annotated` type to specify the required input arguments.
"""
logger.info("Let's start with the native function. Notice that we're add the `@kernel_function` decorator that holds the name of the function as well as an optional description. The input parameters are configured as part of the function's signature, and we use the `Annotated` type to specify the required input arguments.")




class GenerateNumberPlugin:
    """
    Description: Generate a number between a min and a max.
    """

    @kernel_function(
        name="GenerateNumber",
        description="Generate a random number between min and max",
    )
    def generate_number(
        self,
        min: Annotated[int, "the minimum number of paragraphs"],
        max: Annotated[int, "the maximum number of paragraphs"] = 10,
    ) -> Annotated[int, "the output is a number"]:
        """
        Generate a number between min-max
        Example:
            min="4" max="10" => rand(4,8)
        Args:
            min -- The lower limit for the random number generation
            max -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(min, max))
        except ValueError as e:
            logger.debug(f"Invalid input {min} and {max}")
            raise e

generate_number_plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")
generate_number = generate_number_plugin["GenerateNumber"]

"""
Now let's also allow the semantic function to take in additional arguments. In this case, we're going to allow the our CorgiStory function to be written in a specified language. We'll need to provide a `paragraph_count` and a `language`.
"""
logger.info("Now let's also allow the semantic function to take in additional arguments. In this case, we're going to allow the our CorgiStory function to be written in a specified language. We'll need to provide a `paragraph_count` and a `language`.")

prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$paragraph_count}} paragraphs long
- Be written in this language: {{$language}}
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
        InputVariable(name="paragraph_count", description="The number of paragraphs", is_required=True),
        InputVariable(name="language", description="The language of the story", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStory",
    plugin_name="CorgiPlugin",
    prompt_template_config=prompt_template_config,
)

"""
Let's generate a paragraph count.
"""
logger.info("Let's generate a paragraph count.")

async def run_async_code_faf8cd8a():
    async def run_async_code_b9e30d2e():
        result = await generate_number.invoke(kernel, min=1, max=5)
        return result
    result = asyncio.run(run_async_code_b9e30d2e())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_faf8cd8a())
logger.success(format_json(result))
num_paragraphs = result.value
logger.debug(f"Generating a corgi story {num_paragraphs} paragraphs long.")

"""
We can now invoke our corgi_story function using the `kernel` and the keyword arguments `paragraph_count` and `language`.
"""
logger.info("We can now invoke our corgi_story function using the `kernel` and the keyword arguments `paragraph_count` and `language`.")

desired_language = "Spanish"
async def run_async_code_e0533631():
    async def run_async_code_16ea152f():
        story = await corgi_story.invoke(kernel, paragraph_count=num_paragraphs, language=desired_language)
        return story
    story = asyncio.run(run_async_code_16ea152f())
    logger.success(format_json(story))
    return story
story = asyncio.run(run_async_code_e0533631())
logger.success(format_json(story))

logger.debug(f"Generating a corgi story {num_paragraphs} paragraphs long in {desired_language}.")
logger.debug("=====================================================")
logger.debug(story)

"""
## Calling Native Functions within a Semantic Function

One neat thing about the Semantic Kernel is that you can also call native functions from within Prompt Functions!

We will make our CorgiStory semantic function call a native function `GenerateNames` which will return names for our Corgi characters.

We do this using the syntax `{{plugin_name.function_name}}`. You can read more about our prompte templating syntax [here](../../../docs/PROMPT_TEMPLATE_LANGUAGE.md).
"""
logger.info("## Calling Native Functions within a Semantic Function")



class GenerateNamesPlugin:
    """
    Description: Generate character names.
    """

    @kernel_function(description="Generate character names", name="generate_names")
    def generate_names(self) -> str:
        """
        Generate two names.
        Returns:
            str
        """
        names = {"Hoagie", "Hamilton", "Bacon", "Pizza", "Boots", "Shorts", "Tuna"}
        first_name = random.choice(list(names))
        names.remove(first_name)
        second_name = random.choice(list(names))
        return f"{first_name}, {second_name}"

generate_names_plugin = kernel.add_plugin(GenerateNamesPlugin(), plugin_name="GenerateNames")
generate_names = generate_names_plugin["generate_names"]

prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$paragraph_count}} paragraphs long
- Be written in this language: {{$language}}
- The two names of the corgis are {{GenerateNames.generate_names}}
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
    name="corgi-new",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="paragraph_count", description="The number of paragraphs", is_required=True),
        InputVariable(name="language", description="The language of the story", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStoryUpdated",
    plugin_name="CorgiPluginUpdated",
    prompt_template_config=prompt_template_config,
)

async def run_async_code_faf8cd8a():
    async def run_async_code_b9e30d2e():
        result = await generate_number.invoke(kernel, min=1, max=5)
        return result
    result = asyncio.run(run_async_code_b9e30d2e())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_faf8cd8a())
logger.success(format_json(result))
num_paragraphs = result.value

desired_language = "French"
async def run_async_code_e0533631():
    async def run_async_code_16ea152f():
        story = await corgi_story.invoke(kernel, paragraph_count=num_paragraphs, language=desired_language)
        return story
    story = asyncio.run(run_async_code_16ea152f())
    logger.success(format_json(story))
    return story
story = asyncio.run(run_async_code_e0533631())
logger.success(format_json(story))

logger.debug(f"Generating a corgi story {num_paragraphs} paragraphs long in {desired_language}.")
logger.debug("=====================================================")
logger.debug(story)

"""
### Recap

A quick review of what we've learned here:

- We've learned how to create native and prompt functions and register them to the kernel
- We've seen how we can use Kernel Arguments to pass in more custom variables into our prompt
- We've seen how we can call native functions within a prompt.
"""
logger.info("### Recap")

logger.info("\n\n[DONE]", bright=True)