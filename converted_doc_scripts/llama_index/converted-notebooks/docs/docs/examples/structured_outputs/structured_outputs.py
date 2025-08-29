from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from IPython.display import clear_output
from copy import deepcopy
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_parse import LlamaParse
from pprint import pprint
from pydantic import BaseModel, Field
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
# Examples of Structured Data Extraction in LlamaIndex

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/structured_outputs/structured_outputs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you haven't yet read our [structured data extraction tutorial](../../understanding/extraction/index.md), we recommend starting there. This notebook demonstrates some of the techniques introduced in the tutorial.

We start with the simple syntax around LLMs, then move on to how to use it with higher-level modules like a query engine and agent.

A lot of the underlying behavior around structured outputs is powered by our Pydantic Program modules. Check out our [in-depth structured outputs guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/) for more details.
"""
logger.info("# Examples of Structured Data Extraction in LlamaIndex")

# import nest_asyncio

# nest_asyncio.apply()


llm = OllamaFunctionCallingAdapter(
    model="llama3.2", request_timeout=300.0, context_window=4096)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
Settings.llm = llm
Settings.embed_model = embed_model

"""
## 1. Simple Structured Extraction

You can convert any LLM to a "structured LLM" by attaching an output class to it through `as_structured_llm`.

Here we pass a simple `Album` class which contains a list of songs. We can then use the normal LLM endpoints like chat/complete.

**NOTE**: async is supported but streaming is coming soon.
"""
logger.info("## 1. Simple Structured Extraction")


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]


sllm = llm.as_structured_llm(output_cls=Album)
input_msg = ChatMessage.from_str("Generate an example album from The Shining")

"""
#### Sync
"""
logger.info("#### Sync")

output = sllm.chat([input_msg])
output_obj = output.raw

logger.debug(str(output))
logger.debug(output_obj)

"""
#### Async
"""
logger.info("#### Async")

output = sllm.chat([input_msg])
logger.success(format_json(output))
output_obj = output.raw
logger.debug(str(output))

"""
#### Streaming
"""
logger.info("#### Streaming")


stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    clear_output(wait=True)
    plogger.debug(partial_output.raw.dict())

output_obj = partial_output.raw
logger.debug(str(output))

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")


stream_output = sllm.stream_chat([input_msg])
logger.success(format_json(stream_output))
async for partial_output in stream_output:
    clear_output(wait=True)
    plogger.debug(partial_output.raw.dict())

"""
### 1.b Use the `structured_predict` Function

Instead of explicitly doing `llm.as_structured_llm(...)`, every LLM class has a `structured_predict` function which allows you to more easily call the LLM with a prompt template + template variables to return a strutured output in one line of code.
"""
logger.info("### 1.b Use the `structured_predict` Function")


chat_prompt_tmpl = ChatPromptTemplate(
    message_templates=[
        ChatMessage.from_str(
            "Generate an example album from {movie_name}", role="user"
        )
    ]
)

llm = OllamaFunctionCallingAdapter(
    model="llama3.2", request_timeout=300.0, context_window=4096)
album = llm.structured_predict(
    Album, chat_prompt_tmpl, movie_name="Lord of the Rings"
)
album

"""
## 2. Plug into RAG Pipeline

You can also plug this into a RAG pipeline. Below we show structured extraction from an Apple 10K report.
"""
logger.info("## 2. Plug into RAG Pipeline")

# !mkdir data
# !wget "https://s2.q4cdn.com/470004039/files/doc_financials/2021/q4/_10-K-2021-(As-Filed).pdf" -O data/apple_2021_10k.pdf

"""
#### Option 1: Use LlamaParse

You will need an account at https://cloud.llamaindex.ai/ and an API Key to use LlamaParse, our document parser for 10K filings.
"""
logger.info("#### Option 1: Use LlamaParse")


orig_docs = LlamaParse(result_type="text").load_data(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/apple_2021_10k.pdf"
)


def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes


docs = get_page_nodes(orig_docs)
logger.debug(docs[0].get_content())

"""
#### Option 2: Use SimpleDirectoryReader

You can also choose to use the free PDF parser bundled into our `SimpleDirectoryReader`.
"""
logger.info("#### Option 2: Use SimpleDirectoryReader")


"""
#### Build RAG Pipeline, Define Structured Output Schema

We build a RAG pipeline with our trusty VectorStoreIndex and reranker module. We then define the output as a Pydantic model. This allows us to create a structured LLM with the output class attached.
"""
logger.info("#### Build RAG Pipeline, Define Structured Output Schema")


index = VectorStoreIndex(docs)


reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)


class Output(BaseModel):
    """Output containing the response, page numbers, and confidence."""

    response: str = Field(..., description="The answer to the question.")
    page_numbers: List[int] = Field(
        ...,
        description="The page numbers of the sources used to answer this question. Do not include a page number if the context is irrelevant.",
    )
    confidence: float = Field(
        ...,
        description="Confidence value between 0-1 of the correctness of the result.",
    )
    confidence_explanation: str = Field(
        ..., description="Explanation for the confidence score"
    )


sllm = llm.as_structured_llm(output_cls=Output)

"""
#### Run Queries
"""
logger.info("#### Run Queries")

query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[reranker],
    llm=sllm,
    # you can also select other modes like `compact`, `refine`
    response_mode="tree_summarize",
)

response = query_engine.query("Net sales for each product category in 2021")
logger.debug(str(response))

response.response.dict()

logger.info("\n\n[DONE]", bright=True)
