from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.readers.file import FlatReader
from pathlib import Path
from pydantic import BaseModel
from unstructured.partition.html import partition_html
import os
import pandas as pd
import pickle
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/sec_tables/tesla_10q_table.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Joint Tabular/Semantic QA over Tesla 10K

In this example, we show how to ask questions over 10K with understanding of both the unstructured text as well as embedded tables.

We use Unstructured to parse out the tables, and use LlamaIndex recursive retrieval to index/retrieve tables if necessary given the user question.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Joint Tabular/Semantic QA over Tesla 10K")

# %pip install llama-index-readers-file
# %pip install llama-index-llms-ollama

# !pip install llama-index

# %load_ext autoreload
# %autoreload 2


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
## Perform Data Extraction

In these sections we use Unstructured to parse out the table and non-table elements.

### Extract Elements

We use Unstructured to extract table and non-table elements from the 10-K filing.
"""
logger.info("## Perform Data Extraction")

# !wget "https://www.dropbox.com/scl/fi/mlaymdy1ni1ovyeykhhuk/tesla_2021_10k.htm?rlkey=qf9k4zn0ejrbm716j0gg7r802&dl=1" -O tesla_2021_10k.htm
# !wget "https://www.dropbox.com/scl/fi/rkw0u959yb4w8vlzz76sa/tesla_2020_10k.htm?rlkey=tfkdshswpoupav5tqigwz1mp7&dl=1" -O tesla_2020_10k.htm


reader = FlatReader()
docs_2021 = reader.load_data(Path("tesla_2021_10k.htm"))
docs_2020 = reader.load_data(Path("tesla_2020_10k.htm"))


node_parser = UnstructuredElementNodeParser()


if not os.path.exists("2021_nodes.pkl"):
    raw_nodes_2021 = node_parser.get_nodes_from_documents(docs_2021)
    pickle.dump(raw_nodes_2021, open("2021_nodes.pkl", "wb"))
else:
    raw_nodes_2021 = pickle.load(open("2021_nodes.pkl", "rb"))

base_nodes_2021, node_mappings_2021 = node_parser.get_base_nodes_and_mappings(
    raw_nodes_2021
)

example_index_node = [b for b in base_nodes_2021 if isinstance(b, IndexNode)][
    20
]

logger.debug(
    f"\n--------\n{example_index_node.get_content(metadata_mode='all')}\n--------\n"
)
logger.debug(f"\n--------\nIndex ID: {example_index_node.index_id}\n--------\n")
logger.debug(
    f"\n--------\n{node_mappings_2021[example_index_node.index_id].get_content()}\n--------\n"
)

"""
## Setup Recursive Retriever

Now that we've extracted tables and their summaries, we can setup a recursive retriever in LlamaIndex to query these tables.

### Construct Retrievers
"""
logger.info("## Setup Recursive Retriever")


vector_index = VectorStoreIndex(base_nodes_2021)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=1)


recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=node_mappings_2021,
    verbose=True,
)
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

"""
### Run some Queries
"""
logger.info("### Run some Queries")

response = query_engine.query("What was the revenue in 2020?")
logger.debug(str(response))

response = vector_query_engine.query("What was the revenue in 2020?")
logger.debug(str(response))

response = query_engine.query("What were the total cash flows in 2021?")

logger.debug(str(response))

response = vector_query_engine.query("What were the total cash flows in 2021?")
logger.debug(str(response))

response = query_engine.query("What are the risk factors for Tesla?")
logger.debug(str(response))

response = vector_query_engine.query("What are the risk factors for Tesla?")
logger.debug(str(response))

"""
## Try Table Comparisons

In this setting we load in both the 2021 and 2020 10K filings, parse each into a hierarchy of tables/text objects, define a recursive retriever over each, and then compose both with a SubQuestionQueryEngine.

This allows us to execute document comparisons against both.

### Define E2E Recursive Retriever Function
"""
logger.info("## Try Table Comparisons")



def create_recursive_retriever_over_doc(docs, nodes_save_path=None):
    """Big function to go from document path -> recursive retriever."""
    node_parser = UnstructuredElementNodeParser()
    if nodes_save_path is not None and os.path.exists(nodes_save_path):
        raw_nodes = pickle.load(open(nodes_save_path, "rb"))
    else:
        raw_nodes = node_parser.get_nodes_from_documents(docs)
        if nodes_save_path is not None:
            pickle.dump(raw_nodes, open(nodes_save_path, "wb"))

    base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(
        raw_nodes
    )

    vector_index = VectorStoreIndex(base_nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=node_mappings,
        verbose=True,
    )
    query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
    return query_engine, base_nodes

"""
### Create Sub Question Query Engine
"""
logger.info("### Create Sub Question Query Engine")

# import nest_asyncio

# nest_asyncio.apply()



llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

query_engine_2021, nodes_2021 = create_recursive_retriever_over_doc(
    docs_2021, nodes_save_path="2021_nodes.pkl"
)
query_engine_2020, nodes_2020 = create_recursive_retriever_over_doc(
    docs_2020, nodes_save_path="2020_nodes.pkl"
)

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_2021,
        metadata=ToolMetadata(
            name="tesla_2021_10k",
            description=(
                "Provides information about Tesla financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine_2020,
        metadata=ToolMetadata(
            name="tesla_2020_10k",
            description=(
                "Provides information about Tesla financials for year 2020"
            ),
        ),
    ),
]

sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=llm,
    use_async=True,
)

"""
### Try out some Comparisons
"""
logger.info("### Try out some Comparisons")

response = sub_query_engine.query(
    "Can you compare and contrast the cash flow in 2021 with 2020?"
)

logger.debug(str(response))

response = sub_query_engine.query(
    "Can you compare and contrast the R&D expenditures in 2021 vs. 2020?"
)

logger.debug(str(response))

response = sub_query_engine.query(
    "Can you compare and contrast the risk factors in 2021 vs. 2020?"
)

logger.debug(str(response))

"""
#### Try Comparing against Baseline
"""
logger.info("#### Try Comparing against Baseline")

vector_index_2021 = VectorStoreIndex(nodes_2021)
vector_query_engine_2021 = vector_index_2021.as_query_engine(
    similarity_top_k=2
)
vector_index_2020 = VectorStoreIndex(nodes_2020)
vector_query_engine_2020 = vector_index_2020.as_query_engine(
    similarity_top_k=2
)
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine_2021,
        metadata=ToolMetadata(
            name="tesla_2021_10k",
            description=(
                "Provides information about Tesla financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=vector_query_engine_2020,
        metadata=ToolMetadata(
            name="tesla_2020_10k",
            description=(
                "Provides information about Tesla financials for year 2020"
            ),
        ),
    ),
]

base_sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=llm,
    use_async=True,
)

response = base_sub_query_engine.query(
    "Can you compare and contrast the cash flow in 2021 with 2020?"
)
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)