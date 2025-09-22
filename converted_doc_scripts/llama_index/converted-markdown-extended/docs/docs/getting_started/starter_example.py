from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
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
  order: 5
---
# Starter Tutorial (Using Ollama)

This tutorial will show you how to get started building agents with LlamaIndex. We'll start with a basic example and then show how to add RAG (Retrieval-Augmented Generation) capabilities.

<Aside type="tip">
Make sure you've followed the [installation](/python/framework/getting_started/installation) steps first.
</Aside>

<Aside type="tip">
Want to use local models?
If you want to do our starter tutorial using only local models, [check out this tutorial instead](/python/framework/getting_started/starter_example_local).
</Aside>

## Set your Ollama API key

LlamaIndex uses Ollama's `gpt-3.5-turbo` by default. Make sure your API key is available to your code by setting it as an environment variable:
"""
logger.info("# Starter Tutorial (Using Ollama)")

# export OPENAI_API_KEY=XXXXX

# set OPENAI_API_KEY=XXXXX

"""
<Aside type="tip">
If you are using an Ollama-Compatible API, you can use the `OpenAILike` LLM class. You can find more information in the [OpenAILike LLM](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/) integration and [OpenAILike Embeddings](https://docs.llamaindex.ai/en/stable/api_reference/embeddings/openai_like/) integration.
</Aside>

## Basic Agent Example

Let's start with a simple example using an agent that can perform basic multiplication by calling a tool. Create a file called `starter.py`:
"""
logger.info("## Basic Agent Example")



def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(model="llama3.2"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    response = await agent.run("What is 1234 * 4567?")
    logger.success(format_json(response))
    logger.debug(str(response))


if __name__ == "__main__":
    asyncio.run(main())

"""
This will output something like: `The result of \( 1234 \times 4567 \) is \( 5,678,678 \).`

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


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = query_engine.query(query)
    logger.success(format_json(response))
    return str(response)


agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=Ollama(model="llama3.2"),
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
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

"""
<Aside type="tip">
If you used a [vector store integration](/python/framework/module_guides/storing/vector_stores) besides the default, chances are you can just reload from the vector store:
</Aside>

    ```python
    index = VectorStoreIndex.from_vector_store(vector_store)
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
logger.info("## What's Next?")

logger.info("\n\n[DONE]", bright=True)