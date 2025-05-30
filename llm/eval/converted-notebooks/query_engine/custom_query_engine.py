"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/custom_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Defining a Custom Query Engine

You can (and should) define your custom query engines in order to plug into your downstream LlamaIndex workflows, whether you're building RAG, agents, or other applications.

We provide a `CustomQueryEngine` that makes it easy to define your own queries.
"""

"""
## Setup

We first load some sample data and index it.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index


"""
Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

from llama_index.core import PromptTemplate
from jet.llm.ollama.base import Ollama
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("./data//paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

"""
## Building a Custom Query Engine

We build a custom query engine that simulates a RAG pipeline. First perform retrieval, and then synthesis.

To define a `CustomQueryEngine`, you just have to define some initialization parameters as attributes and implement the `custom_query` function.

By default, the `custom_query` can return a `Response` object (which the response synthesizer returns), but it can also just return a string. These are options 1 and 2 respectively.
"""


"""
### Option 1 (`RAGQueryEngine`)
"""


class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj


"""
### Option 2 (`RAGStringQueryEngine`)
"""


qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: Ollama
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)


"""
## Trying it out

We now try it out on our sample data.
"""

"""
### Trying Option 1 (`RAGQueryEngine`)
"""

synthesizer = get_response_synthesizer(response_mode="compact")
query_engine = RAGQueryEngine(
    retriever=retriever, response_synthesizer=synthesizer
)

response = query_engine.query("What did the author do growing up?")

print(str(response))

print(response.source_nodes[0].get_content())

"""
### Trying Option 2 (`RAGStringQueryEngine`)
"""

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

query_engine = RAGStringQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=qa_prompt,
)

response = query_engine.query("What did the author do growing up?")

print(str(response))
