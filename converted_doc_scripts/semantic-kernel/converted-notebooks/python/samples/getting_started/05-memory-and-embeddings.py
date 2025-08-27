import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from dataclasses import dataclass, field
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import (
AzureChatCompletion,
AzureTextEmbedding,
OllamaChatCompletion,
OllamaTextEmbedding,
)
from semantic_kernel.connectors.azure_ai_search import AzureAISearchCollection
from semantic_kernel.connectors.in_memory import InMemoryStore
from semantic_kernel.data.vector import VectorSearchProtocol
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig
from services import Service
from typing import Annotated
from uuid import uuid4
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
# Building Semantic Memory with Embeddings

So far, we've mostly been treating the kernel as a stateless orchestration engine.
We send text into a model API and receive text out.

In a [previous notebook](04-kernel-arguments-chat.ipynb), we used `kernel arguments` to pass in additional
text into prompts to enrich them with more data. This allowed us to create a basic chat experience.

However, if you solely relied on kernel arguments, you would quickly realize that eventually your prompt
would grow so large that you would run into the model's token limit. What we need is a way to persist state
and build both short-term and long-term memory to empower even more intelligent applications.

To do this, we dive into the key concept of `Semantic Memory` in the Semantic Kernel.

Import Semantic Kernel SDK from pypi.org and other dependencies for this example.
"""
logger.info("# Building Semantic Memory with Embeddings")

# %pip install -U semantic-kernel[azure]

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
In order to use memory, we need to instantiate the Kernel with a Memory Storage
and an Embedding service. In this example, we make use of the `VolatileMemoryStore` which can be thought of as a temporary in-memory storage. This memory is not written to disk and is only available during the app session.

When developing your app you will have the option to plug in persistent storage like Azure AI Search, Azure Cosmos Db, PostgreSQL, SQLite, etc. Semantic Memory allows also to index external data sources, without duplicating all the information as you will see further down in this notebook.
"""
logger.info("In order to use memory, we need to instantiate the Kernel with a Memory Storage")


kernel = Kernel()

chat_service_id = "chat"

if selectedService == Service.AzureOllama:

    credential = AzureCliCredential()
    azure_chat_service = AzureChatCompletion(service_id=chat_service_id, credential=credential)
    embedding_gen = AzureTextEmbedding(service_id="embedding", credential=credential)
    kernel.add_service(azure_chat_service)
    kernel.add_service(embedding_gen)
elif selectedService == Service.Ollama:
    oai_chat_service = OllamaChatCompletion(
        service_id=chat_service_id,
    )
    embedding_gen = OllamaTextEmbedding(
        service_id="embedding",
    )
    kernel.add_service(oai_chat_service)
    kernel.add_service(embedding_gen)




@vectorstoremodel(collection_name="simple-model")
@dataclass
class SimpleModel:
    """Simple model to store some text with a ID."""

    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    id: Annotated[str, VectorStoreField("key")] = field(default_factory=lambda: str(uuid4()))
    embedding: Annotated[
        list[float] | str | None, VectorStoreField("vector", dimensions=1536, embedding_generator=embedding_gen)
    ] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = self.text

"""
At its core, Semantic Memory is a set of data structures that allow you to store the meaning of text that come from different data sources, and optionally to store the source text too. These texts can be from the web, e-mail providers, chats, a database, or from your local directory, and are hooked up to the Semantic Kernel through data source connectors.

The texts are embedded or compressed into a vector of floats representing mathematically the texts' contents and meaning. You can read more about embeddings [here](https://aka.ms/sk/embeddings).

### Manually adding memories

Let's create some initial memories "About Me". We can add memories to our `VolatileMemoryStore` by using `SaveInformationAsync`
"""
logger.info("### Manually adding memories")

records = [
    SimpleModel(text="Your budget for 2024 is $100,000"),
    SimpleModel(text="Your savings from 2023 are $50,000"),
    SimpleModel(text="Your investments are $80,000"),
]


in_memory_store = InMemoryStore()

collection = in_memory_store.get_collection(record_type=SimpleModel)
async def run_async_code_3e67b14f():
    await collection.ensure_collection_exists()
    return 
 = asyncio.run(run_async_code_3e67b14f())
logger.success(format_json())
async def run_async_code_32f265fa():
    await collection.upsert(records)
    return 
 = asyncio.run(run_async_code_32f265fa())
logger.success(format_json())

"""
Let's try searching the memory:
"""
logger.info("Let's try searching the memory:")



async def search_memory_examples(collection: VectorSearchProtocol, questions: list[str]) -> None:
    for question in questions:
        logger.debug(f"Question: {question}")
        async def run_async_code_38be008f():
            async def run_async_code_990ee977():
                results = await collection.search(question, top=1)
                return results
            results = asyncio.run(run_async_code_990ee977())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_38be008f())
        logger.success(format_json(results))
        async for result in results.results:
            logger.debug(f"Answer: {result.record.text}")
            logger.debug(f"Score: {result.score}\n")

"""
The default distance metric for the InMemoryCollection is `cosine`, this means that the closer the vectors are, the more similar they are. The `search` method will return the top `3` results by default, but you can change this by passing in the `top` parameter, this is set to `1` here to get only the most relevant result.
"""
logger.info("The default distance metric for the InMemoryCollection is `cosine`, this means that the closer the vectors are, the more similar they are. The `search` method will return the top `3` results by default, but you can change this by passing in the `top` parameter, this is set to `1` here to get only the most relevant result.")

await search_memory_examples(
    collection,
    questions=[
        "What is my budget for 2024?",
        "What are my savings from 2023?",
        "What are my investments?",
    ],
)

"""
Next, we will add a search function to our kernel that will allow us to search the memory store for relevant information. This function will use the `search` method of the `InMemoryStore` to find the most relevant memories based on a query.
"""
logger.info("Next, we will add a search function to our kernel that will allow us to search the memory store for relevant information. This function will use the `search` method of the `InMemoryStore` to find the most relevant memories based on a query.")

func = kernel.add_function(
    plugin_name="memory",
    function=collection.create_search_function(
        function_name="recall",
        description="Searches the memory for relevant information based on the input query.",
    ),
)

"""
Then we will create a prompt that will use the search function to find relevant information in the memory store and return it as part of the prompt.
"""
logger.info("Then we will create a prompt that will use the search function to find relevant information in the memory store and return it as part of the prompt.")



async def setup_chat_with_memory(
    kernel: Kernel,
    service_id: str,
) -> KernelFunction:
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Information about me, from previous conversations:
    - {{recall 'budget by year'}} What is my budget for 2024?
    - {{recall 'savings from previous year'}} What are my savings from 2023?
    - {{recall 'investments'}} What are my investments?

    {{$request}}
    """.strip()

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        execution_settings={
            service_id: kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        },
    )

    return kernel.add_function(
        function_name="chat_with_memory",
        plugin_name="chat",
        prompt_template_config=prompt_template_config,
    )

"""
Now that we've included our memories, let's chat!
"""
logger.info("Now that we've included our memories, let's chat!")

logger.debug("Setting up a chat (with memory!)")
async def run_async_code_9f3ea797():
    async def run_async_code_b2de1142():
        chat_func = await setup_chat_with_memory(kernel, chat_service_id)
        return chat_func
    chat_func = asyncio.run(run_async_code_b2de1142())
    logger.success(format_json(chat_func))
    return chat_func
chat_func = asyncio.run(run_async_code_9f3ea797())
logger.success(format_json(chat_func))

logger.debug("Begin chatting (type 'exit' to exit):\n")
logger.debug(
    "Welcome to the chat bot!\
    \n  Type 'exit' to exit.\
    \n  Try asking a question about your finances (i.e. \"talk to me about my finances\")."
)


async def chat(user_input: str):
    logger.debug(f"User: {user_input}")
    async def run_async_code_879908bc():
        async def run_async_code_ea8b3d95():
            answer = await kernel.invoke(chat_func, request=user_input)
            return answer
        answer = asyncio.run(run_async_code_ea8b3d95())
        logger.success(format_json(answer))
        return answer
    answer = asyncio.run(run_async_code_879908bc())
    logger.success(format_json(answer))
    logger.debug(f"ChatBot:> {answer}")

async def run_async_code_929b3921():
    await chat("What is my budget for 2024?")
    return 
 = asyncio.run(run_async_code_929b3921())
logger.success(format_json())

async def run_async_code_e4ad407b():
    await chat("talk to me about my finances")
    return 
 = asyncio.run(run_async_code_e4ad407b())
logger.success(format_json())

"""
### Adding documents to your memory

Many times in your applications you'll want to bring in external documents into your memory. Let's see how we can do this using our VolatileMemoryStore.

Let's first get some data using some of the links in the Semantic Kernel repo.
"""
logger.info("### Adding documents to your memory")

@vectorstoremodel(collection_name="github-files")
@dataclass
class GitHubFile:
    """
    Model to store GitHub file URLs and their descriptions.
    """

    url: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    key: Annotated[str, VectorStoreField("key")] = field(default_factory=lambda: str(uuid4()))
    embedding: Annotated[
        list[float] | str | None, VectorStoreField("vector", dimensions=1536, embedding_generator=embedding_gen)
    ] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = f"{self.url} {self.text}"

github_files = []
github_files.append(
    GitHubFile(
        url="https://github.com/microsoft/semantic-kernel/blob/main/README.md",
        text="README: Installation, getting started, and how to contribute",
    )
)
github_files.append(
    GitHubFile(
        url="https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/02-running-prompts-from-file.ipynb",
        text="Jupyter notebook describing how to pass prompts from a file to a semantic plugin or function",
    )
)
github_files.append(
    GitHubFile(
        url="https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/00-getting-started.ipynb",
        text="Jupyter notebook describing how to get started with Semantic Kernel",
    )
)

github_memory = in_memory_store.get_collection(record_type=GitHubFile)
async def run_async_code_dea4633b():
    await github_memory.ensure_collection_exists()
    return 
 = asyncio.run(run_async_code_dea4633b())
logger.success(format_json())

async def run_async_code_4066c024():
    await github_memory.upsert(github_files)
    return 
 = asyncio.run(run_async_code_4066c024())
logger.success(format_json())

ask = "I love Jupyter notebooks, how should I get started?"
logger.debug("===========================\n" + "Query: " + ask + "\n")

async def run_async_code_9b20c35f():
    async def run_async_code_adf47b2b():
        memories = await github_memory.search(ask, top=5)
        return memories
    memories = asyncio.run(run_async_code_adf47b2b())
    logger.success(format_json(memories))
    return memories
memories = asyncio.run(run_async_code_9b20c35f())
logger.success(format_json(memories))

async for result in memories.results:
    memory = result.record
    logger.debug(f"Result {memory.key}:")
    logger.debug(f"  URL        : {memory.url}")
    logger.debug(f"  Text       : {memory.text}")
    logger.debug(f"  Relevance  : {result.score}")
    logger.debug()

"""
Now you might be wondering what happens if you have so much data that it doesn't fit into your RAM? That's where you want to make use of an external Vector Database made specifically for storing and retrieving embeddings. Fortunately, semantic kernel makes this easy thanks to an extensive list of available connectors. In the following section, we will connect to an existing Azure AI Search service that we will use as an external Vector Database to store and retrieve embeddings.

_Please note you will need an AzureAI Search api_key or token credential and endpoint for the following example to work properly._
"""
logger.info("Now you might be wondering what happens if you have so much data that it doesn't fit into your RAM? That's where you want to make use of an external Vector Database made specifically for storing and retrieving embeddings. Fortunately, semantic kernel makes this easy thanks to an extensive list of available connectors. In the following section, we will connect to an existing Azure AI Search service that we will use as an external Vector Database to store and retrieve embeddings.")


azs_memory = AzureAISearchCollection(record_type=GitHubFile)
async def run_async_code_31edb60d():
    await azs_memory.ensure_collection_deleted()
    return 
 = asyncio.run(run_async_code_31edb60d())
logger.success(format_json())
async def run_async_code_644ae850():
    await azs_memory.ensure_collection_exists()
    return 
 = asyncio.run(run_async_code_644ae850())
logger.success(format_json())
async def run_async_code_3a1c786d():
    await azs_memory.upsert(github_files)
    return 
 = asyncio.run(run_async_code_3a1c786d())
logger.success(format_json())

"""
The implementation of Semantic Kernel allows to easily swap memory store for another. Here, we will re-use the functions we initially created for `InMemoryStore` with our new external Vector Store leveraging Azure AI Search

Let's now try to query from Azure AI Search! Note that the `score` might be different because the AzureAISearchCollection uses a different default distance metric then InMemoryCollection, if you specify the `distance_function` in the model, you can get the same results as with the InMemoryCollection.
"""
logger.info("The implementation of Semantic Kernel allows to easily swap memory store for another. Here, we will re-use the functions we initially created for `InMemoryStore` with our new external Vector Store leveraging Azure AI Search")

await search_memory_examples(
    azs_memory,
    questions=[
        "What is Semantic Kernel?",
        "How do I get started on it with notebooks?",
        "Where can I find more info on prompts?",
    ],
)

"""
Make sure to cleanup!
"""
logger.info("Make sure to cleanup!")

async def run_async_code_31edb60d():
    await azs_memory.ensure_collection_deleted()
    return 
 = asyncio.run(run_async_code_31edb60d())
logger.success(format_json())

"""
We have laid the foundation which will allow us to store an arbitrary amount of data in an external Vector Store above and beyond what could fit in memory at the expense of a little more latency.
"""
logger.info("We have laid the foundation which will allow us to store an arbitrary amount of data in an external Vector Store above and beyond what could fit in memory at the expense of a little more latency.")

logger.info("\n\n[DONE]", bright=True)