import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OllamaChatCompletion
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
# Groundedness Checking Plugins

For the proper configuration settings and setup, please follow the steps outlined at the beginning of the [first](./00-getting-started.ipynb) getting started notebook.

A well-known problem with large language models (LLMs) is that they make things up. These are sometimes called 'hallucinations' but a safer (and less anthropomorphic) term is 'ungrounded addition' - something in the text which cannot be firmly established. When attempting to establish whether or not something in an LLM response is 'true' we can either check for it in the supplied prompt (this is called 'narrow grounding') or use our general knowledge ('broad grounding'). Note that narrow grounding can lead to things being classified as 'true, but ungrounded.' For example "I live in Switzerland" is **not** _narrowly_ grounded in "I live in Geneva" even though it must be true (it **is** _broadly_ grounded).

In this notebook we run a simple grounding pipeline, to see if a summary text has any ungrounded additions as compared to the original, and use this information to improve the summary text. This can be done in three stages:

1. Make a list of the entities in the summary text
1. Check to see if these entities appear in the original (grounding) text
1. Remove the ungrounded entities from the summary text

What is an 'entity' in this context? In its simplest form, it's a named object such as a person or place (so 'Dean' or 'Seattle'). However, the idea could be a _claim_ which relates concepts (such as 'Dean lives near Seattle'). In this notebook, we will keep to the simpler case of named objects.

Import Semantic Kernel SDK from pypi.org
"""
logger.info("# Groundedness Checking Plugins")

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

Let us define our grounding text:
"""
logger.info("### Configuring the Kernel")

grounding_text = """I am by birth a Genevese, and my family is one of the most distinguished of that republic.
My ancestors had been for many years counsellors and syndics, and my father had filled several public situations
with honour and reputation. He was respected by all who knew him for his integrity and indefatigable attention
to public business. He passed his younger days perpetually occupied by the affairs of his country; a variety
of circumstances had prevented his marrying early, nor was it until the decline of life that he became a husband
and the father of a family.

As the circumstances of his marriage illustrate his character, I cannot refrain from relating them. One of his
most intimate friends was a merchant who, from a flourishing state, fell, through numerous mischances, into poverty.
This man, whose name was Beaufort, was of a proud and unbending disposition and could not bear to live in poverty
and oblivion in the same country where he had formerly been distinguished for his rank and magnificence. Having
paid his debts, therefore, in the most honourable manner, he retreated with his daughter to the town of Lucerne,
where he lived unknown and in wretchedness. My father loved Beaufort with the truest friendship and was deeply
grieved by his retreat in these unfortunate circumstances. He bitterly deplored the false pride which led his friend
to a conduct so little worthy of the affection that united them. He lost no time in endeavouring to seek him out,
with the hope of persuading him to begin the world again through his credit and assistance.

Beaufort had taken effectual measures to conceal himself, and it was ten months before my father discovered his
abode. Overjoyed at this discovery, he hastened to the house, which was situated in a mean street near the Reuss.
But when he entered, misery and despair alone welcomed him. Beaufort had saved but a very small sum of money from
the wreck of his fortunes, but it was sufficient to provide him with sustenance for some months, and in the meantime
he hoped to procure some respectable employment in a merchant's house. The interval was, consequently, spent in
inaction; his grief only became more deep and rankling when he had leisure for reflection, and at length it took
so fast hold of his mind that at the end of three months he lay on a bed of sickness, incapable of any exertion.

His daughter attended him with the greatest tenderness, but she saw with despair that their little fund was
rapidly decreasing and that there was no other prospect of support. But Caroline Beaufort possessed a mind of an
uncommon mould, and her courage rose to support her in her adversity. She procured plain work; she plaited straw
and by various means contrived to earn a pittance scarcely sufficient to support life.

Several months passed in this manner. Her father grew worse; her time was more entirely occupied in attending him;
her means of subsistence decreased; and in the tenth month her father died in her arms, leaving her an orphan and
a beggar. This last blow overcame her, and she knelt by Beaufort's coffin weeping bitterly, when my father entered
the chamber. He came like a protecting spirit to the poor girl, who committed herself to his care; and after the
interment of his friend he conducted her to Geneva and placed her under the protection of a relation. Two years
after this event Caroline became his wife."""

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
## Import the Plugins

We are going to be using the grounding plugin, to check its quality, and remove ungrounded additions:
"""
logger.info("## Import the Plugins")

plugins_directory = "../../../prompt_template_samples/"

groundingSemanticFunctions = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="GroundingPlugin")

"""
We can also extract the individual semantic functions for our use:
"""
logger.info("We can also extract the individual semantic functions for our use:")

entity_extraction = groundingSemanticFunctions["ExtractEntities"]
reference_check = groundingSemanticFunctions["ReferenceCheckEntities"]
entity_excision = groundingSemanticFunctions["ExciseEntities"]

"""
## Calling Individual Semantic Functions

We will start by calling the individual grounding functions in turn, to show their use. For this we need to create a same summary text:
"""
logger.info("## Calling Individual Semantic Functions")

summary_text = """
My father, a respected resident of Milan, was a close friend of a merchant named Beaufort who, after a series of
misfortunes, moved to Zurich in poverty. My father was upset by his friend's troubles and sought him out,
finding him in a mean street. Beaufort had saved a small sum of money, but it was not enough to support him and
his daughter, Mary. Mary procured work to eke out a living, but after ten months her father died, leaving
her a beggar. My father came to her aid and two years later they married when they visited Rome.
"""

summary_text = summary_text.replace("\n", " ").replace("  ", " ")
logger.debug(summary_text)

"""
Some things to note:

- The implied residence of Geneva has been changed to Milan
- Lucerne has been changed to Zurich
- Caroline has been renamed as Mary
- A reference to Rome has been added

The grounding plugin has three stages:

1. Extract entities from a summary text
2. Perform a reference check against the grounding text
3. Excise any entities which failed the reference check from the summary

Now, let us start calling individual semantic functions.

### Extracting the Entities

The first function we need is entity extraction. We are going to take our summary text, and get a list of entities found within it. For this we use `entity_extraction()`:
"""
logger.info("### Extracting the Entities")

async def async_func_0():
    extraction_result = await kernel.invoke(
        entity_extraction,
        input=summary_text,
        topic="people and places",
        example_entities="John, Jane, mother, brother, Paris, Rome",
    )
    return extraction_result
extraction_result = asyncio.run(async_func_0())
logger.success(format_json(extraction_result))

logger.debug(extraction_result)

"""
So we have our list of entities in the summary

### Performing the reference check

We now use the grounding text to see if the entities we found are grounded. We start by adding the grounding text to our context:

With this in place, we can run the reference checking function. This will use both the entity list in the input, and the `reference_context` in the context object itself:
"""
logger.info("### Performing the reference check")

async def async_func_0():
    grounding_result = await kernel.invoke(
        reference_check, input=str(extraction_result.value), reference_context=grounding_text
    )
    return grounding_result
grounding_result = asyncio.run(async_func_0())
logger.success(format_json(grounding_result))

logger.debug(grounding_result)

"""
So we now have a list of ungrounded entities (of course, this list may not be well grounded itself). Let us store this in the context:

### Excising the ungrounded entities

Finally we can remove the ungrounded entities from the summary text:
"""
logger.info("### Excising the ungrounded entities")

async def async_func_0():
    excision_result = await kernel.invoke(
        entity_excision, input=summary_text, ungrounded_entities=str(grounding_result.value)
    )
    return excision_result
excision_result = asyncio.run(async_func_0())
logger.success(format_json(excision_result))

logger.debug(excision_result)

logger.info("\n\n[DONE]", bright=True)