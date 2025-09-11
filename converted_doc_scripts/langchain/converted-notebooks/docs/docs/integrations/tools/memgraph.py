from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langchain_memgraph.tools import QueryMemgraphTool
from langgraph.prebuilt import create_react_agent
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
sidebar_label: Memgraph
---

# MemgraphToolkit

This will help you get started with the Memgraph [toolkit](/docs/concepts/tools/#toolkits). 

Tools within `MemgraphToolkit` are designed for the interaction with the `Memgraph` database.

## Setup

To be able tot follow the steps below, make sure you have a running Memgraph instance on your local host. For more details on how to run Memgraph, take a look at [Memgraph docs](https://memgraph.com/docs/getting-started)

If you want to get automated tracing from runs of individual tools, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# MemgraphToolkit")



"""
### Installation

This toolkit lives in the `langchain-memgraph` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-memgraph

"""
## Instantiation

Now we can instantiate our toolkit:
"""
logger.info("## Instantiation")


db = MemgraphLangChain(url=url, username=username, password=password)

llm = init_chat_model("llama3.2", model_provider="ollama")

toolkit = MemgraphToolkit(
    db=db,  # Memgraph instance
    llm=llm,  # LLM chat model for LLM operations
)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

toolkit.get_tools()

"""
## Invocation

Tools can be individually called by passing an arguments, for QueryMemgraphTool it would be:
"""
logger.info("## Invocation")



tool.invoke({QueryMemgraphTool({"query": "MATCH (n) RETURN n LIMIT 5"})})

"""
## Use within an agent
"""
logger.info("## Use within an agent")


agent_executor = create_react_agent(llm, tools)

example_query = "MATCH (n) RETURN n LIMIT 1"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## API reference

For more details on API visit [Memgraph integration docs](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)