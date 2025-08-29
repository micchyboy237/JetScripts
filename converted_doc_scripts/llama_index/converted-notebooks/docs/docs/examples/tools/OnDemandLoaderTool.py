from jet.logger import CustomLogger
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOllamaFunctionCallingAdapter
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.readers.wikipedia import WikipediaReader
from pydantic import BaseModel
from typing import List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/tools/OnDemandLoaderTool.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OnDemandLoaderTool Tutorial

Our `OnDemandLoaderTool` is a powerful agent tool that allows for "on-demand" data querying from any data source on LlamaHub.

This tool takes in a `BaseReader` data loader, and when called will 1) load data, 2) index data, and 3) query the data.

In this walkthrough, we show how to use the `OnDemandLoaderTool` to convert our Wikipedia data loader into an accessible search tool for a LangChain agent.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OnDemandLoaderTool Tutorial")

# %pip install llama-index-readers-wikipedia

# !pip install llama-index



"""
### Define Tool

We first define the `WikipediaReader`. Note that the `load_data` interface to `WikipediaReader` takes in a list of `pages`. By default, this queries the Wikipedia search endpoint which will autosuggest the relevant pages.

We then wrap it into our `OnDemandLoaderTool`.

By default since we don't specify the `index_cls`, a simple vector store index is initialized.
"""
logger.info("### Define Tool")

reader = WikipediaReader()

tool = OnDemandLoaderTool.from_defaults(
    reader,
    name="Wikipedia Tool",
    description="A tool for loading and querying articles from Wikipedia",
)

"""
#### Testing

We can try running the tool by itself (or as a LangChain tool), just to showcase what the interface is like! 

Note that besides the arguments required for the data loader, the tool also takes in a `query_str` which will be
the query against the index.
"""
logger.info("#### Testing")

tool(["Berlin"], query_str="What's the arts and culture scene in Berlin?")

lc_tool = tool.to_langchain_structured_tool(verbose=True)

lc_tool.run(
    tool_input={
        "pages": ["Berlin"],
        "query_str": "What's the arts and culture scene in Berlin?",
    }
)

"""
### Initialize LangChain Agent

For tutorial purposes, the agent just has access to one tool - the Wikipedia Reader

Note that we need to use Structured Tools from LangChain.
"""
logger.info("### Initialize LangChain Agent")


llm = ChatOllamaFunctionCallingAdapter(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

agent = initialize_agent(
    [lc_tool],
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True,
)

"""
# Now let's run some queries! 

The OnDemandLoaderTool allows the agent to simultaneously 1) load the data from Wikipedia, 2) query that data.
"""
logger.info("# Now let's run some queries!")

agent.run("Tell me about the arts and culture of Berlin")

agent.run("Tell me about the critical reception to The Departed")

logger.info("\n\n[DONE]", bright=True)