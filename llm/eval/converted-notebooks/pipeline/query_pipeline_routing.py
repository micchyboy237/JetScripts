from llama_index.core.query_pipeline import RouterComponent
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import SummaryIndex
from llama_index.core import Document, VectorStoreIndex
from jet.llm.ollama.base import Ollama
from typing import Dict, Any, List, Optional
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core import SimpleDirectoryReader
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Query Pipeline with Routing
#
# Here we showcase our query pipeline with routing.
#
# Routing lets us dynamically choose underlying query pipelines to use given the query and a set of choices.
#
# We offer this as an out-of-the-box abstraction in our [Router Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine.html) guide. Here we show you how to compose a similar pipeline using our Query Pipeline syntax - this allows you to not only define query engines but easily stitch it into a chain/DAG with other modules across the compute graph.

# Load Data
#
# Load in the Paul Graham essay as an example.

# %pip install llama-index-llms-ollama

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt' -O pg_essay.txt


reader = SimpleDirectoryReader(input_files=["pg_essay.txt"])
documents = reader.load_data()

# Setup Query Pipeline with Routing

# Define Modules
#
# We define llm, vector index, summary index, and prompt templates.


hyde_str = """\
Please write a passage to answer the question: {query_str}

Try to include as many key details as possible.

Passage: """
hyde_prompt = PromptTemplate(hyde_str)

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)


summarizer = TreeSummarize(llm=llm)

vector_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=2)

summary_index = SummaryIndex.from_documents(documents)
summary_qrewrite_str = """\
Here's a question:
{query_str}

You are responsible for feeding the question to an agent that given context will try to answer the question.
The context may or may not be relevant. Rewrite the question to highlight the fact that
only some pieces of context (or none) maybe be relevant.
"""
summary_qrewrite_prompt = PromptTemplate(summary_qrewrite_str)
summary_query_engine = summary_index.as_query_engine()

selector = LLMSingleSelector.from_defaults()

# Construct Query Pipelines
#
# Define a query pipeline for vector index, summary index, and join it together with a router.


vector_chain = QueryPipeline(chain=[vector_query_engine])
summary_chain = QueryPipeline(
    chain=[summary_qrewrite_prompt, llm, summary_query_engine], verbose=True
)

choices = [
    "This tool answers specific questions about the document (not summary questions across the document)",
    "This tool answers summary questions about the document (not specific questions)",
]

router_c = RouterComponent(
    selector=selector,
    choices=choices,
    components=[vector_chain, summary_chain],
    verbose=True,
)
qp = QueryPipeline(chain=[router_c], verbose=True)

# Try out Queries

response = qp.run("What did the author do during his time in YC?")
print(str(response))

response = qp.run("What is a summary of this document?")
print(str(response))

logger.info("\n\n[DONE]", bright=True)
