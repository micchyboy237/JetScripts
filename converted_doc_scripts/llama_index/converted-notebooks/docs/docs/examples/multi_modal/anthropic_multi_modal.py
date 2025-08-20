from PIL import Image
from io import BytesIO
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import TextNode
from llama_index.llms.anthropic import Anthropic
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
from pydantic import BaseModel
from typing import List
import matplotlib.pyplot as plt
import os
import qdrant_client
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/anthropic_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal LLM using Anthropic model for image reasoning

Anthropic has recently released its latest Multi modal models: Claude 3 Opus, Claude 3 Sonnet.

1. Claude 3 Opus - claude-3-opus-20240229

2. Claude 3 Sonnet - claude-3-sonnet-20240229

In this notebook, we show how to use Anthropic MultiModal LLM class/abstraction for image understanding/reasoning.

We also show several functions we are now supporting for Anthropic MultiModal LLM:
* `complete` (both sync and async): for a single prompt and list of images
* `chat` (both sync and async): for multiple chat messages
* `stream complete` (both sync and async): for steaming output of complete
* `stream chat` (both sync and async): for steaming output of chat
"""
logger.info("# Multi-Modal LLM using Anthropic model for image reasoning")

# !pip install llama-index-multi-modal-llms-anthropic
# !pip install llama-index-vector-stores-qdrant
# !pip install matplotlib

"""
##  Use Anthropic to understand Images from Local directory
"""
logger.info("##  Use Anthropic to understand Images from Local directory")


# os.environ["ANTHROPIC_API_KEY"] = ""  # Your ANTHROPIC API key here


img = Image.open("../data/images/prometheus_paper_card.png")
plt.imshow(img)


image_documents = SimpleDirectoryReader(
    input_files=["../data/images/prometheus_paper_card.png"]
).load_data()

anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)

response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

logger.debug(response)

"""
## Use `AnthropicMultiModal` to reason images from URLs
"""
logger.info("## Use `AnthropicMultiModal` to reason images from URLs")


image_urls = [
    "https://venturebeat.com/wp-content/uploads/2024/03/Screenshot-2024-03-04-at-12.49.41%E2%80%AFAM.png",
]

img_response = requests.get(image_urls[0])
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

image_url_documents = load_image_urls(image_urls)

response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_url_documents,
)

logger.debug(response)

"""
## Structured Output Parsing from an Image

In this section, we use our multi-modal Pydantic program to generate structured output from an image.
"""
logger.info("## Structured Output Parsing from an Image")


image_documents = SimpleDirectoryReader(
    input_files=["../data/images/ark_email_sample.PNG"]
).load_data()


img = Image.open("../data/images/ark_email_sample.PNG")
plt.imshow(img)



class TickerInfo(BaseModel):
    """List of ticker info."""

    direction: str
    ticker: str
    company: str
    shares_traded: int
    percent_of_total_etf: float


class TickerList(BaseModel):
    """List of stock tickers."""

    fund: str
    tickers: List[TickerInfo]


prompt_template_str = """\
Can you get the stock information in the image \
and return the answer? Pick just one fund.

Make sure the answer is a JSON format corresponding to a Pydantic schema. The Pydantic schema is given below.

"""

anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)


llm_program = MultiModalLLMCompletionProgram.from_defaults(
    output_cls=TickerList,
    image_documents=image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=anthropic_mm_llm,
    verbose=True,
)

response = llm_program()

logger.debug(str(response))

"""
## Index into a Vector Store

In this section we show you how to use Claude 3 to build a RAG pipeline over image data. We first use Claude to extract text from a set of images. We then index the text with an embedding model. Finally, we build a query pipeline over the data.
"""
logger.info("## Index into a Vector Store")

# !wget "https://www.dropbox.com/scl/fi/c1ec6osn0r2ggnitijqhl/mixed_wiki_images_small.zip?rlkey=swwxc7h4qtwlnhmby5fsnderd&dl=1" -O mixed_wiki_images_small.zip
# !unzip mixed_wiki_images_small.zip


anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)


nodes = []
for img_file in Path("mixed_wiki_images_small").glob("*.png"):
    logger.debug(img_file)
    image_documents = SimpleDirectoryReader(input_files=[img_file]).load_data()
    response = anthropic_mm_llm.complete(
        prompt="Describe the images as an alternative text",
        image_documents=image_documents,
    )
    metadata = {"img_file": img_file}
    nodes.append(TextNode(text=str(response), metadata=metadata))



client = qdrant_client.QdrantClient(path="qdrant_mixed_img")

vector_store = QdrantVectorStore(client=client, collection_name="collection")

embed_model = MLXEmbedding()
anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)


query_engine = index.as_query_engine(llm=Anthropic())
response = query_engine.query("Tell me more about the porsche")

logger.debug(str(response))


for n in response.source_nodes:
    display_source_node(n, metadata_mode="all")

logger.info("\n\n[DONE]", bright=True)