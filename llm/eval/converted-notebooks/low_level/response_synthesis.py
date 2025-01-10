from typing import Optional, List
from dataclasses import dataclass
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
import asyncio
import nest_asyncio
from jet.llm.utils import display_jet_source_node
from llama_index.core import PromptTemplate
from jet.llm.ollama.base import Ollama
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
import pinecone
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/response_synthesis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Building Response Synthesis from Scratch
#
# In this tutorial, we show you how to build the "LLM synthesis" component of a RAG pipeline from scratch. Given a set of retrieved Nodes, we'll show you how to synthesize a response even if the retrieved context overflows the context window.
#
# We'll walk through some synthesis strategies:
# - Create and Refine
# - Tree Summarization
#
# We're essentially unpacking our "Response Synthesis" module and exposing that for the user.
#
# We use Ollama as a default LLM but you're free to plug in any LLM you wish.

# Setup
#
# We build an empty Pinecone Index, and define the necessary LlamaIndex wrappers/abstractions so that we can load/index data and get back a vector retriever.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-vector-stores-pinecone
# %pip install llama-index-llms-ollama

# !pip install llama-index

# Load Data

# !mkdir data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"


loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

# Build Pinecone Index, Get Retriever
#
# We use our high-level LlamaIndex abstractions to 1) ingest data into Pinecone, and then 2) get a vector retriever.
#
# Note that we set chunk sizes to 1024.


api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp")

pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)

pinecone_index = pinecone.Index("quickstart")

pinecone_index.delete(deleteAll=True)


vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
splitter = SentenceSplitter(chunk_size=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], storage_context=storage_context
)

retriever = index.as_retriever()

# Given an example question, get a retrieved set of nodes.
#
# We use the retriever to get a set of relevant nodes given a user query. These nodes will then be passed to the response synthesis modules below.

query_str = (
    "Can you tell me about results from RLHF using both model-based and"
    " human-based evaluation?"
)

retrieved_nodes = retriever.retrieve(query_str)

# Building Response Synthesis with LLMs
#
# In this section we'll show how to use LLMs + Prompts to build a response synthesis module.
#
# We'll start from simple strategies (simply stuffing context into a prompt), to more advanced strategies that can handle context overflows.

# 1. Try a Simple Prompt
#
# We first try to synthesize the response using a single input prompt + LLM call.


llm = Ollama(model="text-davinci-003")

qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""
)

# Given an example question, retrieve the set of relevant nodes and try to put it all in the prompt, separated by newlines.

query_str = (
    "Can you tell me about results from RLHF using both model-based and"
    " human-based evaluation?"
)

retrieved_nodes = retriever.retrieve(query_str)


def generate_response(retrieved_nodes, query_str, qa_prompt, llm):
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    fmt_qa_prompt = qa_prompt.format(
        context_str=context_str, query_str=query_str
    )
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt


response, fmt_qa_prompt = generate_response(
    retrieved_nodes, query_str, qa_prompt, llm
)

print(f"*****Response******:\n{response}\n\n")

print(f"*****Formatted Prompt*****:\n{fmt_qa_prompt}\n\n")

# **Problem**: What if we set the top-k retriever to a higher value? The context would overflow!

retriever = index.as_retriever(similarity_top_k=6)
retrieved_nodes = retriever.retrieve(query_str)

response, fmt_qa_prompt = generate_response(
    retrieved_nodes, query_str, qa_prompt, llm
)
print(f"Response (k=5): {response}")

# 2. Try a "Create and Refine" strategy
#
# To deal with context overflows, we can try a strategy where we synthesize a response sequentially through all nodes. Start with the first node and generate an initial response. Then for subsequent nodes, refine the answer using additional context.
#
# This requires us to define a "refine" prompt as well.

refine_prompt = PromptTemplate(
    """\
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer \
(only if needed) with some more context below.
------------
{context_str}
------------
Given the new context, refine the original answer to better answer the query. \
If the context isn't useful, return the original answer.
Refined Answer: \
"""
)


def generate_response_cr(
    retrieved_nodes, query_str, qa_prompt, refine_prompt, llm
):
    """Generate a response using create and refine strategy.

    The first node uses the 'QA' prompt.
    All subsequent nodes use the 'refine' prompt.

    """
    cur_response = None
    fmt_prompts = []
    for idx, node in enumerate(retrieved_nodes):
        print(f"[Node {idx}]")
        display_jet_source_node(node, source_length=2000)
        context_str = node.get_content()
        if idx == 0:
            fmt_prompt = qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
        else:
            fmt_prompt = refine_prompt.format(
                context_str=context_str,
                query_str=query_str,
                existing_answer=str(cur_response),
            )

        cur_response = llm.complete(fmt_prompt)
        fmt_prompts.append(fmt_prompt)

    return str(cur_response), fmt_prompts


response, fmt_prompts = generate_response_cr(
    retrieved_nodes, query_str, qa_prompt, refine_prompt, llm
)

print(str(response))

print(fmt_prompts[0])

print(fmt_prompts[1])

# **Observation**: This is an initial step, but obviously there are inefficiencies. One is the fact that it's quite slow - we make sequential calls. The second piece is that each LLM call is inefficient - we are only inserting a single node, but not "stuffing" the prompt with as much context as necessary.

# 3. Try a Hierarchical Summarization Strategy
#
# Another approach is to try a hierarchical summarization strategy. We generate an answer for each node independently, and then hierarchically combine the answers. This "combine" step could happen once, or for maximum generality can happen recursively until there is one "root" node. That "root" node is then returned as the answer.
#
# We implement this approach below. We have a fixed number of children of 5, so we hierarchically combine 5 children at a time.
#
# **NOTE**: In LlamaIndex this is referred to as "tree_summarize", in LangChain this is referred to as map-reduce.


def combine_results(
    texts,
    query_str,
    qa_prompt,
    llm,
    cur_prompt_list,
    num_children=10,
):
    new_texts = []
    for idx in range(0, len(texts), num_children):
        text_batch = texts[idx: idx + num_children]
        context_str = "\n\n".join([t for t in text_batch])
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        combined_response = llm.complete(fmt_qa_prompt)
        new_texts.append(str(combined_response))
        cur_prompt_list.append(fmt_qa_prompt)

    if len(new_texts) == 1:
        return new_texts[0]
    else:
        return combine_results(
            new_texts, query_str, qa_prompt, llm, num_children=num_children
        )


def generate_response_hs(
    retrieved_nodes, query_str, qa_prompt, llm, num_children=10
):
    """Generate a response using hierarchical summarization strategy.

    Combine num_children nodes hierarchically until we get one root node.

    """
    fmt_prompts = []
    node_responses = []
    for node in retrieved_nodes:
        context_str = node.get_content()
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        node_response = llm.complete(fmt_qa_prompt)
        node_responses.append(node_response)
        fmt_prompts.append(fmt_qa_prompt)

    response_txt = combine_results(
        [str(r) for r in node_responses],
        query_str,
        qa_prompt,
        llm,
        fmt_prompts,
        num_children=num_children,
    )

    return response_txt, fmt_prompts


response, fmt_prompts = generate_response_hs(
    retrieved_nodes, query_str, qa_prompt, llm
)

print(str(response))

# **Observation**: Note that the answer is much more concise than the create-and-refine approach. This is a well-known phemonenon - the reason is because hierarchical summarization tends to compress information at each stage, whereas create and refine encourages adding on more information with each node.
#
# **Observation**: Similar to the above section, there are inefficiencies. We are still generating an answer for each node independently that we can try to optimize away.
#
# Our `ResponseSynthesizer` module handles this!

# 4. [Optional] Let's create an async version of hierarchical summarization!
#
# A pro of the hierarchical summarization approach is that the LLM calls can be parallelized, leading to big speedups in response synthesis.
#
# We implement an async version below. We use asyncio.gather to execute coroutines (LLM calls) for each Node concurrently.


nest_asyncio.apply()


async def acombine_results(
    texts,
    query_str,
    qa_prompt,
    llm,
    cur_prompt_list,
    num_children=10,
):
    fmt_prompts = []
    for idx in range(0, len(texts), num_children):
        text_batch = texts[idx: idx + num_children]
        context_str = "\n\n".join([t for t in text_batch])
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        fmt_prompts.append(fmt_qa_prompt)
        cur_prompt_list.append(fmt_qa_prompt)

    tasks = [llm.acomplete(p) for p in fmt_prompts]
    combined_responses = await asyncio.gather(*tasks)
    new_texts = [str(r) for r in combined_responses]

    if len(new_texts) == 1:
        return new_texts[0]
    else:
        return await acombine_results(
            new_texts, query_str, qa_prompt, llm, num_children=num_children
        )


async def agenerate_response_hs(
    retrieved_nodes, query_str, qa_prompt, llm, num_children=10
):
    """Generate a response using hierarchical summarization strategy.

    Combine num_children nodes hierarchically until we get one root node.

    """
    fmt_prompts = []
    node_responses = []
    for node in retrieved_nodes:
        context_str = node.get_content()
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        fmt_prompts.append(fmt_qa_prompt)

    tasks = [llm.acomplete(p) for p in fmt_prompts]
    node_responses = await asyncio.gather(*tasks)

    response_txt = combine_results(
        [str(r) for r in node_responses],
        query_str,
        qa_prompt,
        llm,
        fmt_prompts,
        num_children=num_children,
    )

    return response_txt, fmt_prompts

response, fmt_prompts = await agenerate_response_hs(
    retrieved_nodes, query_str, qa_prompt, llm
)

print(str(response))

# Let's put it all together!
#
# Let's define a simple query engine that can be initialized with a retriever, prompt, llm etc. And have it implement a simple `query` function. We also implement an async version, can be used if you completed part 4 above!
#
# **NOTE**: We skip subclassing our own `QueryEngine` abstractions. This is a big TODO to make it more easily sub-classable!


@dataclass
class Response:
    response: str
    source_nodes: Optional[List] = None

    def __str__(self):
        return self.response


class MyQueryEngine:
    """My query engine.

    Uses the tree summarize response synthesis module by default.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        qa_prompt: PromptTemplate,
        llm: LLM,
        num_children=10,
    ) -> None:
        self._retriever = retriever
        self._qa_prompt = qa_prompt
        self._llm = llm
        self._num_children = num_children

    def query(self, query_str: str):
        retrieved_nodes = self._retriever.retrieve(query_str)
        response_txt, _ = generate_response_hs(
            retrieved_nodes,
            query_str,
            self._qa_prompt,
            self._llm,
            num_children=self._num_children,
        )
        response = Response(response_txt, source_nodes=retrieved_nodes)
        return response

    async def aquery(self, query_str: str):
        retrieved_nodes = await self._retriever.aretrieve(query_str)
        response_txt, _ = await agenerate_response_hs(
            retrieved_nodes,
            query_str,
            self._qa_prompt,
            self._llm,
            num_children=self._num_children,
        )
        response = Response(response_txt, source_nodes=retrieved_nodes)
        return response


query_engine = MyQueryEngine(retriever, qa_prompt, llm, num_children=10)

response = query_engine.query(query_str)

print(str(response))

response = query_engine.query(query_str)

print(str(response))

logger.info("\n\n[DONE]", bright=True)
