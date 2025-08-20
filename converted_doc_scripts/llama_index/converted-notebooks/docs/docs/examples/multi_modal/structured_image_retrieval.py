import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Image
from io import BytesIO
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from typing import Optional
import matplotlib.pyplot as plt
import os
import qdrant_client
import random
import requests
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
# Semi-structured Image Retrieval

In this notebook we show you how to perform semi-structured retrieval over images.

Given a set of images, we can infer structured outputs from them using Gemini Pro Vision.

We can then index these structured outputs in a vector database. We then take full advantage of semantic search + metadata filter capabilities with **auto-retrieval**: this allows us to ask both structured and semantic questions over this data!

(An alternative is to put this data into a SQL database, letting you do text-to-SQL. These techniques are quite closely related).
"""
logger.info("# Semi-structured Image Retrieval")

# %pip install llama-index-multi-modal-llms-gemini
# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-embeddings-gemini
# %pip install llama-index-llms-gemini

# !pip install llama-index 'google-generativeai>=0.3.0' matplotlib qdrant_client

"""
## Setup

### Get Google API Key
"""
logger.info("## Setup")


GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

"""
### Download Images

We download the full SROIE v2 dataset from Kaggle [here](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2).

This dataset consists of scanned receipt images. We ignore the ground-truth labels for now, and use the test set images to test out Gemini's capabilities for structured output extraction.

### Get Image Files

Now that the images are downloaded, we can get a list of the file names.
"""
logger.info("### Download Images")


def get_image_files(
    dir_path, sample: Optional[int] = 10, shuffle: bool = False
):
    dir_path = Path(dir_path)
    image_paths = []
    for image_path in dir_path.glob("*.jpg"):
        image_paths.append(image_path)

    random.shuffle(image_paths)
    if sample:
        return image_paths[:sample]
    else:
        return image_paths

image_files = get_image_files("SROIE2019/test/img", sample=100)

"""
## Use Gemini to extract structured outputs

Here we use Gemini to extract structured outputs.
1. Define a ReceiptInfo pydantic class that captures the structured outputs we want to extract. We extract fields like `company`, `date`, `total`, and also `summary`.
2. Define a `pydantic_gemini` function which will convert input documents into a response.

### Define a ReceiptInfo pydantic class
"""
logger.info("## Use Gemini to extract structured outputs")



class ReceiptInfo(BaseModel):
    company: str = Field(..., description="Company name")
    date: str = Field(..., description="Date field in DD/MM/YYYY format")
    address: str = Field(..., description="Address")
    total: float = Field(..., description="total amount")
    currency: str = Field(
        ..., description="Currency of the country (in abbreviations)"
    )
    summary: str = Field(
        ...,
        description="Extracted text summary of the receipt, including items purchased, the type of store, the location, and any other notable salient features (what does the purchase seem to be for?).",
    )

"""
### Define a `pydantic_gemini` function
"""
logger.info("### Define a `pydantic_gemini` function")


prompt_template_str = """\
    Can you summarize the image and return a response \
    with the following JSON format: \
"""


async def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY, model_name="models/gemini-pro-vision"
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    async def run_async_code_aaacf9f6():
        async def run_async_code_1ebb6866():
            response = await llm_program.acall()
            return response
        response = asyncio.run(run_async_code_1ebb6866())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_aaacf9f6())
    logger.success(format_json(response))
    return response

"""
### Run over images
"""
logger.info("### Run over images")



async def aprocess_image_file(image_file):
    logger.debug(f"Image file: {image_file}")
    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    async def run_async_code_46aa69a1():
        async def run_async_code_6586c6ca():
            output = await pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)
            return output
        output = asyncio.run(run_async_code_6586c6ca())
        logger.success(format_json(output))
        return output
    output = asyncio.run(run_async_code_46aa69a1())
    logger.success(format_json(output))
    return output


async def aprocess_image_files(image_files):
    """Process metadata on image files."""

    new_docs = []
    tasks = []
    for image_file in image_files:
        task = aprocess_image_file(image_file)
        tasks.append(task)

    async def run_async_code_95e81316():
        async def run_async_code_c92a0d2e():
            outputs = await run_jobs(tasks, show_progress=True, workers=5)
            return outputs
        outputs = asyncio.run(run_async_code_c92a0d2e())
        logger.success(format_json(outputs))
        return outputs
    outputs = asyncio.run(run_async_code_95e81316())
    logger.success(format_json(outputs))
    return outputs

async def run_async_code_1cb7e9e3():
    async def run_async_code_74001706():
        outputs = await aprocess_image_files(image_files)
        return outputs
    outputs = asyncio.run(run_async_code_74001706())
    logger.success(format_json(outputs))
    return outputs
outputs = asyncio.run(run_async_code_1cb7e9e3())
logger.success(format_json(outputs))

outputs[4]

"""
### Convert Structured Representation to `TextNode` objects

Node objects are the core units that are indexed in vector stores in LlamaIndex. We define a simple converter function to map the `ReceiptInfo` objects to `TextNode` objects.
"""
logger.info("### Convert Structured Representation to `TextNode` objects")



def get_nodes_from_objs(
    objs: List[ReceiptInfo], image_files: List[str]
) -> TextNode:
    """Get nodes from objects."""
    nodes = []
    for image_file, obj in zip(image_files, objs):
        node = TextNode(
            text=obj.summary,
            metadata={
                "company": obj.company,
                "date": obj.date,
                "address": obj.address,
                "total": obj.total,
                "currency": obj.currency,
                "image_file": str(image_file),
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes

nodes = get_nodes_from_objs(outputs, image_files)

logger.debug(nodes[0].get_content(metadata_mode="all"))

"""
### Index these nodes in vector stores
"""
logger.info("### Index these nodes in vector stores")


client = qdrant_client.QdrantClient(path="qdrant_gemini")

vector_store = QdrantVectorStore(client=client, collection_name="collection")

Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
Settings.llm = (Gemini(api_key=GOOGLE_API_KEY),)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

"""
## Define Auto-Retriever

Now we can setup our auto-retriever, which can perform semi-structured queries: structured queries through inferring metadata filters, along with semantic search.

We setup our schema definition capturing the receipt info which is fed into the prompt.
"""
logger.info("## Define Auto-Retriever")



vector_store_info = VectorStoreInfo(
    content_info="Receipts",
    metadata_info=[
        MetadataInfo(
            name="company",
            description="The name of the store",
            type="string",
        ),
        MetadataInfo(
            name="address",
            description="The address of the store",
            type="string",
        ),
        MetadataInfo(
            name="date",
            description="The date of the purchase (in DD/MM/YYYY format)",
            type="string",
        ),
        MetadataInfo(
            name="total",
            description="The final amount",
            type="float",
        ),
        MetadataInfo(
            name="currency",
            description="The currency of the country the purchase was made (abbreviation)",
            type="string",
        ),
    ],
)


retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
    verbose=True,
)



def display_response(nodes: List[TextNode]):
    """Display response."""
    for node in nodes:
        logger.debug(node.get_content(metadata_mode="all"))
        display(Image(filename=node.metadata["image_file"], width=200))

"""
## Run Some Queries

Let's try out different types of queries!
"""
logger.info("## Run Some Queries")

nodes = retriever.retrieve(
    "Tell me about some restaurant orders of noodles with total < 25"
)
display_response(nodes)

nodes = retriever.retrieve("Tell me about some grocery purchases")
display_response(nodes)

logger.info("\n\n[DONE]", bright=True)