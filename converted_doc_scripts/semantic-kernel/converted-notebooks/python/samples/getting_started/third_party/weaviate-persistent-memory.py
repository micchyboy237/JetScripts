import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion, OllamaTextEmbedding
from semantic_kernel.connectors.memory.weaviate import WeaviateMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.prompt_template import PromptTemplateConfig
from services import Service
from weaviate.embedded import EmbeddedOptions
import os
import shutil
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Introduction

This notebook shows how to replace the `VolatileMemoryStore` memory storage used in a [previous notebook](./06-memory-and-embeddings.ipynb) with a `WeaviateMemoryStore`.

`WeaviateMemoryStore` is an example of a persistent (i.e. long-term) memory store backed by the Weaviate vector database.

### Configuring the Kernel

Let's get started with the necessary configuration to run Semantic Kernel. For Notebooks, we require a `.env` file with the proper settings for the model you use. Create a new file named `.env` and place it in this directory. Copy the contents of the `.env.example` file from this directory and paste it into the `.env` file that you just created.

**NOTE: Please make sure to include `GLOBAL_LLM_SERVICE` set to either Ollama, AzureOllama, or HuggingFace in your .env file. If this setting is not included, the Service will default to AzureOllama.**

#### Option 1: using Ollama

Add your [Ollama Key](https://platform.openai.com/docs/overview) key to your `.env` file (org Id only if you have multiple orgs):

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

For more advanced configuration, please follow the steps outlined in the [setup guide](./CONFIGURING_THE_KERNEL.md).

# About Weaviate

[Weaviate](https://weaviate.io/) is an open-source vector database designed to scale seamlessly into billions of data objects. This implementation supports hybrid search out-of-the-box (meaning it will perform better for keyword searches).

You can run Weaviate in 5 ways:

- **SaaS** – with [Weaviate Cloud Services (WCS)](https://weaviate.io/pricing).

  WCS is a fully managed service that takes care of hosting, scaling, and updating your Weaviate instance. You can try it out for free with a sandbox that lasts for 14 days.

  To set up a SaaS Weaviate instance with WCS:

  1.  Navigate to [Weaviate Cloud Console](https://console.weaviate.cloud/).
  2.  Register or sign in to your WCS account.
  3.  Create a new cluster with the following settings:
      - `Subscription Tier` – Free sandbox for a free trial, or contact [hello@weaviate.io](mailto:hello@weaviate.io) for other options.
      - `Cluster name` – a unique name for your cluster. The name will become part of the URL used to access this instance.
      - `Enable Authentication?` – Enabled by default. This will generate a static API key that you can use to authenticate.
  4.  Wait for a few minutes until your cluster is ready. You will see a green tick ✔️ when it's done. Copy your cluster URL.

- **Hybrid SaaS**

  > If you need to keep your data on-premise for security or compliance reasons, Weaviate also offers a Hybrid SaaS option: Weaviate runs within your cloud instances, but the cluster is managed remotely by Weaviate. This gives you the benefits of a managed service without sending data to an external party.

  The Weaviate Hybrid SaaS is a custom solution. If you are interested in this option, please reach out to [hello@weaviate.io](mailto:hello@weaviate.io).

- **Self-hosted** – with a Docker container

  To set up a Weaviate instance with Docker:

  1. [Install Docker](https://docs.docker.com/engine/install/) on your local machine if it is not already installed.
  2. [Install the Docker Compose Plugin](https://docs.docker.com/compose/install/)
  3. Download a `docker-compose.yml` file with this `curl` command:

     ```
     curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.19.6"
     ```

     Alternatively, you can use Weaviate's docker compose [configuration tool](https://weaviate.io/developers/weaviate/installation/docker-compose) to generate your own `docker-compose.yml` file.

  4. Run `docker compose up -d` to spin up a Weaviate instance.

     > To shut it down, run `docker compose down`.

- **Self-hosted** – with a Kubernetes cluster

  To configure a self-hosted instance with Kubernetes, follow Weaviate's [documentation](https://weaviate.io/developers/weaviate/installation/kubernetes).|

- **Embedded** - start a weaviate instance right from your application code using the client library

  This code snippet shows how to instantiate an embedded weaviate instance and upload a document:

  ```python

  client = weaviate.Client(
    embedded_options=EmbeddedOptions()
  )

  data_obj = {
    "name": "Chardonnay",
    "description": "Goes with fish"
  }

  client.data_object.create(data_obj, "Wine")
  ```

  Refer to the [documentation](https://weaviate.io/developers/weaviate/installation/embedded) for more details about this deployment method.

# Setup
"""
logger.info("# Introduction")

# %pip install -U semantic-kernel[weaviate]

__version__

"""
## OS-specific notes:

- if you run into SSL errors when connecting to Ollama on macOS, see this issue for a [potential solution](https://github.com/microsoft/semantic-kernel/issues/627#issuecomment-1580912248)
- on Windows, you may need to run Docker Desktop as administrator

First, we instantiate the Weaviate memory store. Uncomment ONE of the options below, depending on how you want to use Weaviate:

- from a Docker instance
- from WCS
- directly from the client (embedded Weaviate), which works on Linux only at the moment
"""
logger.info("## OS-specific notes:")



store = WeaviateMemoryStore()
store.client.schema.delete_all()

"""
Then, we register the memory store to the kernel:
"""
logger.info("Then, we register the memory store to the kernel:")


selectedService = Service.Ollama


kernel = Kernel()

chat_service_id = "chat"
if selectedService == Service.Ollama:
    oai_chat_service = OllamaChatCompletion(
        service_id=chat_service_id,
        ai_model_id="gpt-3.5-turbo",
    )
    embedding_gen = OllamaTextEmbedding(ai_model_id="text-embedding-ada-002")
    kernel.add_service(oai_chat_service)
    kernel.add_service(embedding_gen)

memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

"""
# Manually adding memories

Let's create some initial memories "About Me". We can add memories to our weaviate memory store by using `save_information`
"""
logger.info("# Manually adding memories")

collection_id = "generic"


async def populate_memory(memory: SemanticTextMemory) -> None:
    async def run_async_code_e9d6089b():
        await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
        return 
     = asyncio.run(run_async_code_e9d6089b())
    logger.success(format_json())
    async def run_async_code_24ea5af2():
        await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
        return 
     = asyncio.run(run_async_code_24ea5af2())
    logger.success(format_json())
    async def run_async_code_8859963e():
        await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")
        return 
     = asyncio.run(run_async_code_8859963e())
    logger.success(format_json())

async def run_async_code_91f67f7f():
    await populate_memory(memory)
    return 
 = asyncio.run(run_async_code_91f67f7f())
logger.success(format_json())

"""
Searching is done through `search`:
"""
logger.info("Searching is done through `search`:")

async def search_memory_examples(memory: SemanticTextMemory) -> None:
    questions = ["What is my budget for 2024?", "What are my savings from 2023?", "What are my investments?"]

    for question in questions:
        logger.debug(f"Question: {question}")
        async def run_async_code_9133fbf1():
            async def run_async_code_5843a8ae():
                result = await memory.search(collection_id, question)
                return result
            result = asyncio.run(run_async_code_5843a8ae())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_9133fbf1())
        logger.success(format_json(result))
        logger.debug(f"Answer: {result[0].text}\n")

async def run_async_code_50dc1bc2():
    await search_memory_examples(memory)
    return 
 = asyncio.run(run_async_code_50dc1bc2())
logger.success(format_json())

"""
Here's how to use the weaviate memory store in a chat application:
"""
logger.info("Here's how to use the weaviate memory store in a chat application:")



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
        plugin_name="TextMemoryPlugin",
        prompt_template_config=prompt_template_config,
    )

async def chat(kernel: Kernel, chat_func: KernelFunction) -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        logger.debug("\n\nExiting chat...")
        return False
    except EOFError:
        logger.debug("\n\nExiting chat...")
        return False

    if user_input == "exit":
        logger.debug("\n\nExiting chat...")
        return False

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
    return True

logger.debug("Populating memory...")
async def run_async_code_91f67f7f():
    await populate_memory(memory)
    return 
 = asyncio.run(run_async_code_91f67f7f())
logger.success(format_json())

logger.debug("Asking questions... (manually)")
async def run_async_code_50dc1bc2():
    await search_memory_examples(memory)
    return 
 = asyncio.run(run_async_code_50dc1bc2())
logger.success(format_json())

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
chatting = True
while chatting:
    async def run_async_code_67415c0e():
        async def run_async_code_5910d94f():
            chatting = await chat(kernel, chat_func)
            return chatting
        chatting = asyncio.run(run_async_code_5910d94f())
        logger.success(format_json(chatting))
        return chatting
    chatting = asyncio.run(run_async_code_67415c0e())
    logger.success(format_json(chatting))

"""
# Adding documents to your memory

Create a dictionary to hold some files. The key is the hyperlink to the file and the value is the file's content:
"""
logger.info("# Adding documents to your memory")

github_files = {}
github_files["https://github.com/microsoft/semantic-kernel/blob/main/README.md"] = (
    "README: Installation, getting started, and how to contribute"
)
github_files[
    "https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/02-running-prompts-from-file.ipynb"
] = "Jupyter notebook describing how to pass prompts from a file to a semantic plugin or function"
github_files["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/00-getting-started.ipynb"] = (
    "Jupyter notebook describing how to get started with the Semantic Kernel"
)
github_files["https://github.com/microsoft/semantic-kernel/tree/main/samples/plugins/ChatPlugin/ChatGPT"] = (
    "Sample demonstrating how to create a chat plugin interfacing with ChatGPT"
)
github_files[
    "https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/SemanticKernel/Memory/Volatile/VolatileMemoryStore.cs"
] = "C# class that defines a volatile embedding store"

"""
Use `save_reference` to save the file:
"""
logger.info("Use `save_reference` to save the file:")

COLLECTION = "SKGitHub"

logger.debug("Adding some GitHub file URLs and their descriptions to a volatile Semantic Memory.")
for index, (entry, value) in enumerate(github_files.items()):
    await memory.save_reference(
        collection=COLLECTION,
        description=value,
        text=value,
        external_id=entry,
        external_source_name="GitHub",
    )
    logger.debug("  URL {} saved".format(index))

"""
Use `search` to ask a question:
"""
logger.info("Use `search` to ask a question:")

ask = "I love Jupyter notebooks, how should I get started?"
logger.debug("===========================\n" + "Query: " + ask + "\n")

async def run_async_code_1821303a():
    async def run_async_code_034fa610():
        memories = await memory.search(COLLECTION, ask, limit=5, min_relevance_score=0.77)
        return memories
    memories = asyncio.run(run_async_code_034fa610())
    logger.success(format_json(memories))
    return memories
memories = asyncio.run(run_async_code_1821303a())
logger.success(format_json(memories))

for index, memory in enumerate(memories):
    logger.debug(f"Result {index}:")
    logger.debug("  URL:     : " + memory.id)
    logger.debug("  Title    : " + memory.description)
    logger.debug("  Relevance: " + str(memory.relevance))
    logger.debug()

logger.info("\n\n[DONE]", bright=True)