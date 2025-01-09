from program_ollama import OllamaPydanticProgram
from llama_index.core.extractors import PydanticProgramExtractor
from typing import List
from pydantic import BaseModel, Field
from llama_index.core.response.notebook_utils import (
    display_source_node,
    display_response,
)
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode
from jet.llm.ollama.base import Ollama
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Metadata Extraction
#
# In this notebook we will demonstrate following:
#
# 1. RAG using Metadata Extractors.
# 2. Extract Metadata using PydanticProgram.

# Installation

# !pip install llama-index
# !pip install llama_index-readers-web


nest_asyncio.apply()


# Setup API Key

# os.environ["OPENAI_API_KEY"] = "sk-..."

# Define LLM


llm = Ollama(temperature=0.1, model="llama3.2",
             request_timeout=300.0, context_window=4096, max_tokens=512)
Settings.llm = llm

# Node Parser and Metadata Extractors


node_parser = TokenTextSplitter(
    separator=" ", chunk_size=256, chunk_overlap=128
)

question_extractor = QuestionsAnsweredExtractor(
    questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
)

# Load Data


reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])

print(docs[0].get_content())

# Nodes

orig_nodes = node_parser.get_nodes_from_documents(docs)

print(orig_nodes[20:28][3].get_content(metadata_mode="all"))

# Question Extractor on Nodes

nodes_1 = node_parser.get_nodes_from_documents(docs)[20:28]
nodes_1 = question_extractor(nodes_1)

print(nodes_1[3].get_content(metadata_mode="all"))

# Build Indices


index0 = VectorStoreIndex(orig_nodes)
index1 = VectorStoreIndex(orig_nodes[:20] + nodes_1 + orig_nodes[28:])

# Query Engines

query_engine0 = index0.as_query_engine(similarity_top_k=1)
query_engine1 = index1.as_query_engine(similarity_top_k=1)

# Querying

query_str = (
    "Can you describe metrics for evaluating text generation quality, compare"
    " them, and tell me about their downsides"
)

response0 = query_engine0.query(query_str)
logger.log("Query:", query_str, colors=["WHITE", "INFO"])
logger.newline()
logger.log("Score:", f"{response0.node.score:.2f}",
           colors=["WHITE", "SUCCESS"])
logger.log("File:", response0.node.metadata, colors=["WHITE", "DEBUG"])

response1 = query_engine1.query(query_str)
logger.log("Query:", query_str, colors=["WHITE", "INFO"])
logger.newline()
logger.log("Score:", f"{response1.node.score:.2f}",
           colors=["WHITE", "SUCCESS"])
logger.log("File:", response1.node.metadata, colors=["WHITE", "DEBUG"])

# Extract Metadata Using PydanticProgramExtractor
#
# PydanticProgramExtractor enables extracting an entire Pydantic object using an LLM.
#
# This approach allows for extracting multiple entities in a single LLM call, offering an advantage over using a single metadata extractor.


# Setup the Pydantic Model¶
#
# Here we define a basic structured schema that we want to extract. It contains:
#
# Entities: unique entities in a text chunk
# Summary: a concise summary of the text chunk


class NodeMetadata(BaseModel):
    """Node metadata."""

    entities: List[str] = Field(
        ..., description="Unique entities in this text chunk."
    )
    summary: str = Field(
        ..., description="A concise summary of this text chunk."
    )

# Setup the Extractor¶


EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""

openai_program = OllamaPydanticProgram.from_defaults(
    output_cls=NodeMetadata,
    prompt_template_str="{input}",
    extract_template_str=EXTRACT_TEMPLATE_STR,
)

metadata_extractor = PydanticProgramExtractor(
    program=openai_program, input_key="input", show_progress=True
)

# Extract metadata from the node

extract_metadata = metadata_extractor.extract(orig_nodes[0:1])

extract_metadata

metadata_nodes = metadata_extractor.process_nodes(orig_nodes[0:1])

metadata_nodes

logger.info("\n\n[DONE]", bright=True)
