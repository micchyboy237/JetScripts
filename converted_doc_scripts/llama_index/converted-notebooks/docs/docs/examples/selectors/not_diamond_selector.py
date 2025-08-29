from jet.logger import CustomLogger
from llama_index.core import (
SimpleDirectoryReader,
VectorStoreIndex,
SummaryIndex,
Settings,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.selectors.notdiamond.base import NotDiamondSelector
from notdiamond import NotDiamond
from notdiamond import NotDiamond, Metric
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/RouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Using Not Diamond to Select LLMs For Indexes
In this tutorial, we demonstrate how to use a router query engine with a selector powered by [Not Diamond](https://www.notdiamond.ai). You can automatically route a query to one of several available LLMs, which will then select the best index for your needs.

### Setup
"""
logger.info("# Using Not Diamond to Select LLMs For Indexes")

# %pip install -q llama-index-llms-anthropic llama-index-llms-ollama

# !pip install -q llama-index notdiamond

# import nest_asyncio

# nest_asyncio.apply()

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Routing Queries With Not Diamond
"""
logger.info("## Routing Queries With Not Diamond")


# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["NOTDIAMOND_API_KEY"] = "sk-..."

"""
### Create Indexes
"""
logger.info("### Create Indexes")


documents = SimpleDirectoryReader("data/paul_graham").load_data()
nodes = Settings.node_parser.get_nodes_from_documents(documents)

vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)
query_text = "What was Paul Graham's role at Yahoo?"

"""
### Set up Tools for the QueryEngine
"""
logger.info("### Set up Tools for the QueryEngine")

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)

"""
### Create a NotDiamondSelector and RouterQueryEngine
"""
logger.info("### Create a NotDiamondSelector and RouterQueryEngine")


client = NotDiamond(
    api_key=os.environ["NOTDIAMOND_API_KEY"],
    llm_configs=["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620"],
)
preference_id = client.create_preference_id()
client.preference_id = preference_id

nd_selector = NotDiamondSelector(client=client)

query_engine = RouterQueryEngine(
    selector=nd_selector,
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

"""
### Use Not Diamond to Query Indexes

Once we've set up our indexes and query engine, we can submit queries as usual.
"""
logger.info("### Use Not Diamond to Query Indexes")

response = query_engine.query(
    "Please summarize Paul Graham's working experience."
)
logger.debug(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
logger.debug(str(response))

"""
## Using NotDiamondSelector as a standalone selector

As with LlamaIndex's built-in selectors, you can also use the `NotDiamondSelector` to select an index.
"""
logger.info("## Using NotDiamondSelector as a standalone selector")



choices = [
    ToolMetadata(
        name="vector_index",
        description="Great for asking questions about recipes.",
    ),
    ToolMetadata(
        name="list_index", description="Great for summarizing recipes."
    ),
]

llm_configs = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620"]
nd_client = NotDiamond(
    api_key=os.environ["NOTDIAMOND_API_KEY"],
    llm_configs=llm_configs,
    preference_id=preference_id,
)
preference_id = nd_client.create_preference_id()
nd_client.preference_id = preference_id
nd_selector = NotDiamondSelector(client=nd_client)

nd_result = nd_selector.select(
    choices, query="What is the summary of this recipe for deviled eggs?"
)
logger.debug(nd_result)

metric = Metric("accuracy")
score = metric.feedback(
    session_id=nd_result.session_id,
    llm_config=nd_result.llm,
    value=1,
)

logger.info("\n\n[DONE]", bright=True)