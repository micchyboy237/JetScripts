import asyncio
from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.llm.ollama.adapters.ollama_llama_index_llm_question_generator_adapter import LLMQuestionGenerator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import UnstructuredReader
# from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from pathlib import Path
import os
import shutil


# Setup logger and settings
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp"
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/Chatbot_SEC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# How to Build a Chatbot

LlamaIndex serves as a bridge between your data and Language Learning Models (LLMs), providing a toolkit that enables you to establish a query interface around your data for a variety of tasks, such as question-answering and summarization.

In this tutorial, we'll walk you through building a context-augmented chatbot using a [Data Agent](https://gpt-index.readthedocs.io/en/stable/core_modules/agent_modules/agents/root.html). This agent, powered by LLMs, is capable of intelligently executing tasks over your data. The end result is a chatbot agent equipped with a robust set of data interface tools provided by LlamaIndex to answer queries about your data.

**Note**: This tutorial builds upon initial work on creating a query interface over SEC 10-K filings - [check it out here](https://medium.com/@jerryjliu98/how-unstructured-and-llamaindex-can-help-bring-the-power-of-llms-to-your-own-data-3657d063e30d).

### Context

In this guide, we’ll build a "10-K Chatbot" that uses raw UBER 10-K HTML filings from Dropbox. Users can interact with the chatbot to ask questions related to the 10-K filings.

### Preparation
"""
logger.info("# How to Build a Chatbot")

# %pip install llama-index-readers-file
# %pip install llama-index-embeddings-ollama
# %pip install llama-index-agent-openai
# %pip install llama-index-llms-ollama
# %pip install llama-index-question-gen-openai
# %pip install unstructured


# os.environ["OPENAI_API_KEY"] = "sk-..."


"""
### Ingest Data

Let's first download the raw 10-k files, from 2019-2022.
"""
logger.info("### Ingest Data")

# !mkdir data
# !wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O data/UBER.zip
# !unzip data/UBER.zip -d data

"""
To parse the HTML files into formatted text, we use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library. Thanks to [LlamaHub](https://llamahub.ai/), we can directly integrate with Unstructured, allowing conversion of any text into a Document format that LlamaIndex can ingest.

First we install the necessary packages:

Then we can use the `UnstructuredReader` to parse the HTML files into a list of `Document` objects.
"""
logger.info(
    "To parse the HTML files into formatted text, we use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library. Thanks to [LlamaHub](https://llamahub.ai/), we can directly integrate with Unstructured, allowing conversion of any text into a Document format that LlamaIndex can ingest.")


years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"{DATA_DIR}/UBER/UBER_{year}.html"), split_documents=False
    )
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

"""
### Setting up Vector Indices for each year

We first setup a vector index for each year. Each vector index allows us
to ask questions about the 10-K filing of a given year.

We build each index and save it to disk.
"""
logger.info("### Setting up Vector Indices for each year")


index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"{OUTPUT_DIR}/storage/{year}")

"""
To load an index from disk, do the following
"""
logger.info("To load an index from disk, do the following")


index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{OUTPUT_DIR}/storage/{year}"
    )
    cur_index = load_index_from_storage(
        storage_context,
    )
    index_set[year] = cur_index

"""
### Setting up a Sub Question Query Engine to Synthesize Answers Across 10-K Filings

Since we have access to documents of 4 years, we may not only want to ask questions regarding the 10-K document of a given year, but ask questions that require analysis over all 10-K filings.

To address this, we can use a [Sub Question Query Engine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/sub_question_query_engine.html). It decomposes a query into subqueries, each answered by an individual vector index, and synthesizes the results to answer the overall query.

LlamaIndex provides some wrappers around indices (and query engines) so that they can be used by query engines and agents. First we define a `QueryEngineTool` for each vector index.
Each tool has a name and a description; these are what the LLM agent sees to decide which tool to choose.
"""
logger.info(
    "### Setting up a Sub Question Query Engine to Synthesize Answers Across 10-K Filings")


# Create query engine tools and sub-question query engine
individual_query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=index_set[year].as_query_engine(),
        name=f"vector_index_{year}",
        description=(
            "useful for when you want to answer queries about the"
            f" {year} SEC 10-K for Uber"
        ),
    )
    for year in years
]

"""
Now we can create the Sub Question Query Engine, which will allow us to synthesize answers across the 10-K filings. We pass in the `individual_query_engine_tools` we defined above.
"""
logger.info("Now we can create the Sub Question Query Engine, which will allow us to synthesize answers across the 10-K filings. We pass in the `individual_query_engine_tools` we defined above.")


query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=Settings.llm,
    question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
)

"""
### Setting up the Chatbot Agent

We use a LlamaIndex Data Agent to setup the outer chatbot agent, which has access to a set of Tools. Specifically, we will use an FunctionAgent, that takes advantage of Ollama API function calling. We want to use the separate Tools we defined previously for each index (corresponding to a given year), as well as a tool for the sub question query engine we defined above.

First we define a `QueryEngineTool` for the sub question query engine:
"""
logger.info("### Setting up the Chatbot Agent")

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="sub_question_query_engine",
    description=(
        "useful for when you want to answer queries that require analyzing"
        " multiple SEC 10-K documents for Uber"
    ),
)

"""
Then, we combine the Tools we defined above into a single list of tools for the agent:
"""
logger.info(
    "Then, we combine the Tools we defined above into a single list of tools for the agent:")

tools = individual_query_engine_tools + [query_engine_tool]

"""
Finally, we call `FunctionAgent` to create the agent, passing in the list of tools we defined above.
"""
logger.info(
    "Finally, we call `FunctionAgent` to create the agent, passing in the list of tools we defined above.")


agent = FunctionAgent(
    tools=tools, llm=OllamaFunctionCallingAdapter(model="llama3.2"))

"""
### Testing the Agent

We can now test the agent with various queries.

If we test it with a simple "hello" query, the agent does not use any Tools.
"""


# Test functions using a single event loop
async def run_tests():
    logger.info("### Testing the Agent")

    # Test 1: Simple "hello" query
    ctx = Context(agent)
    logger.info("Testing with 'hi, i am bob'")
    response = await agent.run("hi, i am bob", ctx=ctx)
    logger.success(format_json(response))

    # Test 2: Risk factors for 2020
    ctx = Context(agent)  # Create new context
    logger.info("Testing risk factors for 2020")
    response = await agent.run(
        "What were some of the biggest risk factors in 2020 for Uber?", ctx=ctx
    )
    logger.success(format_json(response))

    # Test 3: Cross-year comparison
    ctx = Context(agent)  # Create new context
    cross_query_str = (
        "Compare/contrast the risk factors described in the Uber 10-K across"
        " years. Give answer in bullet points."
    )
    logger.info("Testing cross-year risk factors comparison")
    response = await agent.run(cross_query_str, ctx=ctx)
    logger.success(format_json(response))


"""
### Setting up the Chatbot Loop

Now that we have the chatbot setup, it only takes a few more steps to setup a basic interactive loop to chat with our SEC-augmented chatbot!
"""
logger.info("### Setting up the Chatbot Loop")


# Interactive chat loop
async def chat_loop():
    ctx = Context(agent)
    while True:
        text_input = input("User: ")
        if text_input.lower() == "exit":
            break
        response = await agent.run(text_input, ctx=ctx)
        logger.debug(f"Agent:")
        logger.success(format_json(response))

# Run all tests and chat loop in a single event loop
if __name__ == "__main__":
    async def main():
        await run_tests()
        logger.info("Starting interactive chat loop")
        await chat_loop()
        logger.info("\n\n[DONE]", bright=True)

    asyncio.run(main())
