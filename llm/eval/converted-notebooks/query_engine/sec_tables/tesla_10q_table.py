"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/sec_tables/tesla_10q_table.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Joint Tabular/Semantic QA over Tesla 10K

In this example, we show how to ask questions over 10K with understanding of both the unstructured text as well as embedded tables.

We use Unstructured to parse out the tables, and use LlamaIndex recursive retrieval to index/retrieve tables if necessary given the user question.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-readers-file
# %pip install llama-index-llms-ollama

# !pip install llama-index

# %load_ext autoreload
# %autoreload 2


from jet.llm.ollama import Ollama
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import nest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
import pickle
import os
from llama_index.core.node_parser import UnstructuredElementNodeParser
from pathlib import Path
from llama_index.readers.file import FlatReader
from pydantic import BaseModel
from unstructured.partition.html import partition_html
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
## Perform Data Extraction

In these sections we use Unstructured to parse out the table and non-table elements.
"""

"""
### Extract Elements

We use Unstructured to extract table and non-table elements from the 10-K filing.
"""

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

print(
    f"\n--------\n{example_index_node.get_content(
        metadata_mode='all')}\n--------\n"
)
print(f"\n--------\nIndex ID: {example_index_node.index_id}\n--------\n")
print(
    f"\n--------\n{
        node_mappings_2021[example_index_node.index_id].get_content()}\n--------\n"
)

"""
## Setup Recursive Retriever

Now that we've extracted tables and their summaries, we can setup a recursive retriever in LlamaIndex to query these tables.
"""

"""
### Construct Retrievers
"""


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

response = query_engine.query("What was the revenue in 2020?")
print(str(response))

response = vector_query_engine.query("What was the revenue in 2020?")
print(str(response))

response = query_engine.query("What were the total cash flows in 2021?")

print(str(response))

response = vector_query_engine.query("What were the total cash flows in 2021?")
print(str(response))

response = query_engine.query("What are the risk factors for Tesla?")
print(str(response))

response = vector_query_engine.query("What are the risk factors for Tesla?")
print(str(response))

"""
## Try Table Comparisons

In this setting we load in both the 2021 and 2020 10K filings, parse each into a hierarchy of tables/text objects, define a recursive retriever over each, and then compose both with a SubQuestionQueryEngine.

This allows us to execute document comparisons against both.
"""

"""
### Define E2E Recursive Retriever Function
"""


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


nest_asyncio.apply()


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

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

response = sub_query_engine.query(
    "Can you compare and contrast the cash flow in 2021 with 2020?"
)

print(str(response))

response = sub_query_engine.query(
    "Can you compare and contrast the R&D expenditures in 2021 vs. 2020?"
)

print(str(response))

response = sub_query_engine.query(
    "Can you compare and contrast the risk factors in 2021 vs. 2020?"
)

print(str(response))

"""
#### Try Comparing against Baseline
"""

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
print(str(response))
