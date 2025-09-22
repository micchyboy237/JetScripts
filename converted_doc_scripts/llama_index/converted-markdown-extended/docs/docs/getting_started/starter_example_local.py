from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from jet.logger import logger
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import asyncio
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
sidebar:
  order: 6
---
# Starter Tutorial (Using Local LLMs)

This tutorial will show you how to get started building agents with LlamaIndex. We'll start with a basic example and then show how to add RAG (Retrieval-Augmented Generation) capabilities.


We will use [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) as our embedding model and `llama3.1 8B` served through `Ollama`.

<Aside type="tip">
Make sure you've followed the [installation](/python/framework/getting_started/installation) steps first.
</Aside>

## Setup

Ollama is a tool to help you get set up with LLMs locally with minimal setup.

Follow the [README](https://github.com/jmorganca/ollama) to learn how to install it.

To download the Llama3 model just do `ollama pull llama3.1`.

**NOTE**: You will need a machine with at least ~32GB of RAM.

As explained in our [installation guide](/python/framework/getting_started/installation), `llama-index` is actually a collection of packages. To run Ollama and Huggingface, we will need to install those integrations:
"""
logger.info("# Starter Tutorial (Using Local LLMs)")

pip install llama-index-llms-ollama llama-index-embeddings-huggingface

"""
The package names spell out the imports, which is very helpful for remembering how to import them or install them!
"""
logger.info("The package names spell out the imports, which is very helpful for remembering how to import them or install them!")


"""
More integrations are all listed on [https://llamahub.ai](https://llamahub.ai).

## Basic Agent Example

Let's start with a simple example using an agent that can perform basic multiplication by calling a tool. Create a file called `starter.py`:
"""
logger.info("## Basic Agent Example")



def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(
        model="llama3.1",
        request_timeout=360.0,
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    response = await agent.run("What is 1234 * 4567?")
    logger.success(format_json(response))
    logger.debug(str(response))


if __name__ == "__main__":
    asyncio.run(main())

"""
This will output something like: `The answer to 1234 * 4567 is: 5,618,916.`

What happened is:

- The agent was given a question: `What is 1234 * 4567?`
- Under the hood, this question, plus the schema of the tools (name, docstring, and arguments) were passed to the LLM
- The agent selected the `multiply` tool and wrote the arguments to the tool
- The agent received the result from the tool and interpolated it into the final response

<Aside type="tip">
As you can see, we are using `async` python functions. Many LLMs and models support async calls, and using async code is recommended to improve performance of your application. To learn more about async code and python, we recommend this [short section on async + python](/python/framework/getting_started/async_python).
</Aside>

## Adding Chat History

The `AgentWorkflow` is also able to remember previous messages. This is contained inside the `Context` of the `AgentWorkflow`.

If the `Context` is passed in, the agent will use it to continue the conversation.
"""
logger.info("## Adding Chat History")


ctx = Context(agent)

response = await agent.run("My name is Logan", ctx=ctx)
logger.success(format_json(response))
response = await agent.run("What is my name?", ctx=ctx)
logger.success(format_json(response))

"""
## Adding RAG Capabilities

Now let's enhance our agent by adding the ability to search through documents. First, let's get some example data using our terminal:
"""
logger.info("## Adding RAG Capabilities")

mkdir data
wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -O data/paul_graham_essay.txt

"""
Your directory structure should look like this now:

<pre>
├── starter.py
└── data
    └── paul_graham_essay.txt
</pre>

Now we can create a tool for searching through documents using LlamaIndex. By default, our `VectorStoreIndex` will use a `text-embedding-ada-002` embeddings from Ollama to embed and retrieve the text.

Our modified `starter.py` should look like this:
"""
logger.info("Your directory structure should look like this now:")


Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=360.0,
    context_window=8000,
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine(
)


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = query_engine.query(query)
    logger.success(format_json(response))
    return str(response)


agent = AgentWorkflow.from_tools_or_functions(
    [multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


async def main():
    response = await agent.run(
            "What did the author do in college? Also, what's 7 * 8?"
        )
    logger.success(format_json(response))
    logger.debug(response)


if __name__ == "__main__":
    asyncio.run(main())

"""
The agent can now seamlessly switch between using the calculator and searching through documents to answer questions.

## Storing the RAG Index

To avoid reprocessing documents every time, you can persist the index to disk:
"""
logger.info("## Storing the RAG Index")

index.storage_context.persist("storage")


storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(
    storage_context,
)
query_engine = index.as_query_engine(
)

"""
<Aside type="tip">
If you used a [vector store integration](/python/framework/module_guides/storing/vector_stores) besides the default, chances are you can just reload from the vector store:
</Aside>

    ```python
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        # it's important to use the same embed_model as the one used to build the index
        # embed_model=Settings.embed_model,
    )
    ```

## What's Next?

This is just the beginning of what you can do with LlamaIndex agents! You can:

- Add more tools to your agent
- Use different LLMs
- Customize the agent's behavior using system prompts
- Add streaming capabilities
- Implement human-in-the-loop workflows
- Use multiple agents to collaborate on tasks

Some helpful next links:

- See more advanced agent examples in our [Agent documentation](/python/framework/understanding/agent)
- Learn more about [high-level concepts](/python/framework/getting_started/concepts)
- Explore how to [customize things](/python/framework/getting_started/faq)
- Check out the [component guides](/python/framework/module_guides)
"""
logger.info("# it's important to use the same embed_model as the one used to build the index")

logger.info("\n\n[DONE]", bright=True)