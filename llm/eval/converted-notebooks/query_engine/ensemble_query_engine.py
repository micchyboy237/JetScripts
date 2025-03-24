"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/ensemble_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Ensemble Query Engine Guide

Oftentimes when building a RAG application there are different query pipelines you need to experiment with (e.g. top-k retrieval, keyword search, knowledge graphs).

Thought: what if we could try a bunch of strategies at once, and have the LLM 1) rate the relevance of each query, and 2) synthesize the results?

This guide showcases this over the Great Gatsby. We do ensemble retrieval over different chunk sizes and also different indices.

**NOTE**: Please also see our closely-related [Ensemble Retrieval Guide](https://gpt-index.readthedocs.io/en/stable/examples/retrievers/ensemble_retrieval.html)!
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Setup
"""


from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core import PromptTemplate
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Settings
from jet.llm.ollama.base import Ollama
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
nest_asyncio.apply()

"""
## Download Data
"""

# !wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/gatsby/gatsby_full.txt' -O 'gatsby_full.txt'

"""
## Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""


documents = SimpleDirectoryReader(
    input_files=["./gatsby_full.txt"]
).load_data()

"""
## Define Query Engines
"""


Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.chunk_size = 1024

nodes = Settings.node_parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


keyword_index = SimpleKeywordTableIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
vector_index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)


QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question. If the answer is not in the context, inform "
    "the user that you can't answer the question - DO NOT MAKE UP AN ANSWER.\n"
    "In addition to returning the answer, also return a relevance score as to "
    "how relevant the answer is to the question. "
    "Question: {query_str}\n"
    "Answer (including relevance score): "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

keyword_query_engine = keyword_index.as_query_engine(
    text_qa_template=QA_PROMPT
)
vector_query_engine = vector_index.as_query_engine(text_qa_template=QA_PROMPT)

response = vector_query_engine.query(
    "Describe and summarize the interactions between Gatsby and Daisy"
)

print(response)

response = keyword_query_engine.query(
    "Describe and summarize the interactions between Gatsby and Daisy"
)

print(response)

"""
## Define Router Query Engine
"""


keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for answering questions about this essay",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for answering questions about this essay",
)


TREE_SUMMARIZE_PROMPT_TMPL = (
    "Context information from multiple sources is below. Each source may or"
    " may not have \na relevance score attached to"
    " it.\n---------------------\n{context_str}\n---------------------\nGiven"
    " the information from multiple sources and their associated relevance"
    " scores (if provided) and not prior knowledge, answer the question. If"
    " the answer is not in the context, inform the user that you can't answer"
    " the question.\nQuestion: {query_str}\nAnswer: "
)

tree_summarize = TreeSummarize(
    summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL)
)

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        keyword_tool,
        vector_tool,
    ],
    summarizer=tree_summarize,
)

"""
## Experiment with Queries
"""

response = query_engine.query(
    "Describe and summarize the interactions between Gatsby and Daisy"
)
print(response)

response.source_nodes

response = query_engine.query(
    "What part of his past is Gatsby trying to recapture?"
)
print(response)
