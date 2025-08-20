from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import (
QuestionsAnsweredExtractor,
)
from llama_index.core.extractors import PydanticProgramExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.response.notebook_utils import (
display_source_node,
display_response,
)
from llama_index.core.schema import MetadataMode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.program.openai import MLXPydanticProgram
from llama_index.readers.web import SimpleWebPageReader
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Metadata Extraction

In this notebook we will demonstrate following:

1. RAG using Metadata Extractors.
2. Extract Metadata using PydanticProgram.

## Installation
"""
logger.info("# Metadata Extraction")

# !pip install llama-index
# !pip install llama_index-readers-web

# import nest_asyncio

# nest_asyncio.apply()


"""
## Setup API Key
"""
logger.info("## Setup API Key")

# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Define LLM
"""
logger.info("## Define LLM")


llm = MLX(temperature=0.1, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_tokens=512)
Settings.llm = llm

"""
## Node Parser and Metadata Extractors
"""
logger.info("## Node Parser and Metadata Extractors")


node_parser = TokenTextSplitter(
    separator=" ", chunk_size=256, chunk_overlap=128
)

question_extractor = QuestionsAnsweredExtractor(
    questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
)

"""
## Load Data
"""
logger.info("## Load Data")


reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])

logger.debug(docs[0].get_content())

"""
## Nodes
"""
logger.info("## Nodes")

orig_nodes = node_parser.get_nodes_from_documents(docs)

logger.debug(orig_nodes[20:28][3].get_content(metadata_mode="all"))

"""
## Question Extractor on Nodes
"""
logger.info("## Question Extractor on Nodes")

nodes_1 = node_parser.get_nodes_from_documents(docs)[20:28]
nodes_1 = question_extractor(nodes_1)

logger.debug(nodes_1[3].get_content(metadata_mode="all"))

"""
## Build Indices
"""
logger.info("## Build Indices")


index0 = VectorStoreIndex(orig_nodes)
index1 = VectorStoreIndex(orig_nodes[:20] + nodes_1 + orig_nodes[28:])

"""
## Query Engines
"""
logger.info("## Query Engines")

query_engine0 = index0.as_query_engine(similarity_top_k=1)
query_engine1 = index1.as_query_engine(similarity_top_k=1)

"""
## Querying
"""
logger.info("## Querying")

query_str = (
    "Can you describe metrics for evaluating text generation quality, compare"
    " them, and tell me about their downsides"
)

response0 = query_engine0.query(query_str)
response1 = query_engine1.query(query_str)

display_response(
    response0, source_length=1000, show_source=True, show_source_metadata=True
)

display_response(
    response1, source_length=1000, show_source=True, show_source_metadata=True
)

"""
## Extract Metadata Using PydanticProgramExtractor

PydanticProgramExtractor enables extracting an entire Pydantic object using an LLM.

This approach allows for extracting multiple entities in a single LLM call, offering an advantage over using a single metadata extractor.
"""
logger.info("## Extract Metadata Using PydanticProgramExtractor")


"""
### Setup the Pydantic Model¶

Here we define a basic structured schema that we want to extract. It contains:

Entities: unique entities in a text chunk
Summary: a concise summary of the text chunk
"""
logger.info("### Setup the Pydantic Model¶")

class NodeMetadata(BaseModel):
    """Node metadata."""

    entities: List[str] = Field(
        ..., description="Unique entities in this text chunk."
    )
    summary: str = Field(
        ..., description="A concise summary of this text chunk."
    )

"""
### Setup the Extractor¶
"""
logger.info("### Setup the Extractor¶")


EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""

openai_program = MLXPydanticProgram.from_defaults(
    output_cls=NodeMetadata,
    prompt_template_str="{input}",
    extract_template_str=EXTRACT_TEMPLATE_STR,
)

metadata_extractor = PydanticProgramExtractor(
    program=openai_program, input_key="input", show_progress=True
)

"""
### Extract metadata from the node
"""
logger.info("### Extract metadata from the node")

extract_metadata = metadata_extractor.extract(orig_nodes[0:1])

extract_metadata

metadata_nodes = metadata_extractor.process_nodes(orig_nodes[0:1])

metadata_nodes

logger.info("\n\n[DONE]", bright=True)