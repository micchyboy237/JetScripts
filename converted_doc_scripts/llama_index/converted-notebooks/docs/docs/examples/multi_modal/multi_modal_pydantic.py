from PIL import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import MLXMultiModal
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.multi_modal_llms.replicate.base import (
REPLICATE_MULTI_MODAL_LLM_MODELS,
)
from pathlib import Path
from pydantic import BaseModel
import matplotlib.pyplot as plt
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/multi_modal_pydantic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal GPT4V Pydantic Program

In this notebook, we show you how to generate `structured data` with new MLX GPT4V API via LlamaIndex. The user just needs to specify a Pydantic object.

We also compared several Large Vision models for this task:
* GPT4-V
* Fuyu-8B
* MiniGPT-4
* CogVLM
* Llava-14B

## Download Image Locally
"""
logger.info("# Multi-Modal GPT4V Pydantic Program")

# %pip install llama-index-multi-modal-llms-openai
# %pip install llama-index-multi-modal-llms-replicate


# OPENAI_API_KEY = "sk-<your-openai-api-token>"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

REPLICATE_API_TOKEN = ""  # Your Relicate API token here
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


input_image_path = Path("restaurant_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)

# !wget "https://docs.google.com/uc?export=download&id=1GlqcNJhGGbwLKjJK1QJ_nyswCTQ2K2Fq" -O ./restaurant_images/fried_chicken.png

"""
## Initialize Pydantic Class for Restaurant
"""
logger.info("## Initialize Pydantic Class for Restaurant")



class Restaurant(BaseModel):
    """Data model for an restaurant."""

    restaurant: str
    food: str
    discount: str
    price: str
    rating: str
    review: str

"""
## Load MLX GPT4V Multi-Modal LLM Model
"""
logger.info("## Load MLX GPT4V Multi-Modal LLM Model")


image_documents = SimpleDirectoryReader("./restaurant_images").load_data()

openai_mm_llm = MLXMultiModal(
#     model="qwen3-1.7b-4bit", api_key=OPENAI_API_KEY, max_new_tokens=1000
)

"""
## Plot the image
"""
logger.info("## Plot the image")


imageUrl = "./restaurant_images/fried_chicken.png"
image = Image.open(imageUrl).convert("RGB")

plt.figure(figsize=(16, 5))
plt.imshow(image)

"""
## Using Multi-Modal Pydantic Program to generate structured data from GPT4V Output for Restaurant Image
"""
logger.info("## Using Multi-Modal Pydantic Program to generate structured data from GPT4V Output for Restaurant Image")


prompt_template_str = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""
openai_program = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Restaurant),
    image_documents=image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=openai_mm_llm,
    verbose=True,
)

response = openai_program()
for res in response:
    logger.debug(res)

"""
## Test Pydantic for MiniGPT-4, Fuyu-8B, LLaVa-13B, CogVLM models
"""
logger.info("## Test Pydantic for MiniGPT-4, Fuyu-8B, LLaVa-13B, CogVLM models")


prompt_template_str = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""


def pydantic_replicate(
    model_name, output_class, image_documents, prompt_template_str
):
    mm_llm = ReplicateMultiModal(
        model=REPLICATE_MULTI_MODAL_LLM_MODELS[model_name],
        temperature=0.1,
        max_new_tokens=1000,
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=mm_llm,
        verbose=True,
    )

    response = llm_program()
    logger.debug(f"Model: {model_name}")
    for res in response:
        logger.debug(res)

"""
### Using Fuyu-8B for Pydantic Strucured Output
"""
logger.info("### Using Fuyu-8B for Pydantic Strucured Output")

pydantic_replicate("fuyu-8b", Restaurant, image_documents, prompt_template_str)

"""
### Using LLaVa-13B for Pydantic Strucured Output
"""
logger.info("### Using LLaVa-13B for Pydantic Strucured Output")

pydantic_replicate(
    "llava-13b", Restaurant, image_documents, prompt_template_str
)

"""
### Using MiniGPT-4 for Pydantic Strucured Output
"""
logger.info("### Using MiniGPT-4 for Pydantic Strucured Output")

pydantic_replicate(
    "minigpt-4", Restaurant, image_documents, prompt_template_str
)

"""
### Using CogVLM for Pydantic Strucured Output
"""
logger.info("### Using CogVLM for Pydantic Strucured Output")

pydantic_replicate("cogvlm", Restaurant, image_documents, prompt_template_str)

"""
`Observation`:
* Only GPT4-V works pretty well for this image pydantic task
* Other vision model can output part fields

## Change to Amazon Product Example
### Download the Amazon Product Image Screenshot
"""
logger.info("## Change to Amazon Product Example")

input_image_path = Path("amazon_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)

# !wget "https://docs.google.com/uc?export=download&id=1p1Y1qAoM68eC4sAvvHaiJyPhdUZS0Gqb" -O ./amazon_images/amazon.png

"""
## Initialize the Amazon Product Pydantic Class
"""
logger.info("## Initialize the Amazon Product Pydantic Class")



class Product(BaseModel):
    """Data model for a Amazon Product."""

    title: str
    category: str
    discount: str
    price: str
    rating: str
    review: str
    description: str
    inventory: str

"""
### Plot the Image
"""
logger.info("### Plot the Image")

imageUrl = "./amazon_images/amazon.png"
image = Image.open(imageUrl).convert("RGB")

plt.figure(figsize=(16, 5))
plt.imshow(image)

"""
## Using Multi-Modal Pydantic Program to generate structured data from GPT4V Output for Amazon Product Image
"""
logger.info("## Using Multi-Modal Pydantic Program to generate structured data from GPT4V Output for Amazon Product Image")

amazon_image_documents = SimpleDirectoryReader("./amazon_images").load_data()

prompt_template_str = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""
openai_program_amazon = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Product),
    image_documents=amazon_image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=openai_mm_llm,
    verbose=True,
)

response = openai_program_amazon()
for res in response:
    logger.debug(res)

"""
## Test Pydantic for MiniGPT-4, Fuyu-8B, LLaVa-13B, CogVLM models

### Using Fuyu-8B for Pydantic Strucured Output
"""
logger.info("## Test Pydantic for MiniGPT-4, Fuyu-8B, LLaVa-13B, CogVLM models")

pydantic_replicate(
    "fuyu-8b", Product, amazon_image_documents, prompt_template_str
)

"""
### Using MiniGPT-4 for Pydantic Strucured Output
"""
logger.info("### Using MiniGPT-4 for Pydantic Strucured Output")

pydantic_replicate(
    "minigpt-4", Product, amazon_image_documents, prompt_template_str
)

"""
### Using CogVLM-4 for Pydantic Strucured Output
"""
logger.info("### Using CogVLM-4 for Pydantic Strucured Output")

pydantic_replicate(
    "cogvlm", Product, amazon_image_documents, prompt_template_str
)

"""
### Using LlaVa-13B for Pydantic Strucured Output
"""
logger.info("### Using LlaVa-13B for Pydantic Strucured Output")

pydantic_replicate(
    "llava-13b", Product, amazon_image_documents, prompt_template_str
)

"""
`Observation`:
* Only GPT4v, Llava-13B and GogVLM output desired fields
* Among those 3 models, GPT4V get the most accurate results. Llava-13B and CogVLM got wrong price.

## Initialize the Instagram Ads Pydantic Class and compare performance of different Multi-Modal LLMs
"""
logger.info("## Initialize the Instagram Ads Pydantic Class and compare performance of different Multi-Modal LLMs")

input_image_path = Path("instagram_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)

# !wget "https://docs.google.com/uc?export=download&id=12ZpBBFkYu-jzz1iz356U5kMikn4uN9ww" -O ./instagram_images/jordan.png



class InsAds(BaseModel):
    """Data model for a Ins Ads."""

    account: str
    brand: str
    product: str
    category: str
    discount: str
    price: str
    comments: str
    review: str
    description: str


imageUrl = "./instagram_images/jordan.png"
image = Image.open(imageUrl).convert("RGB")

plt.figure(figsize=(16, 5))
plt.imshow(image)

ins_image_documents = SimpleDirectoryReader("./instagram_images").load_data()

prompt_template_str = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""
openai_program_ins = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(InsAds),
    image_documents=ins_image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=openai_mm_llm,
    verbose=True,
)


response = openai_program_ins()
for res in response:
    logger.debug(res)

pydantic_replicate("fuyu-8b", InsAds, ins_image_documents, prompt_template_str)

pydantic_replicate(
    "llava-13b", InsAds, ins_image_documents, prompt_template_str
)

pydantic_replicate("cogvlm", InsAds, ins_image_documents, prompt_template_str)

pydantic_replicate(
    "minigpt-4", InsAds, ins_image_documents, prompt_template_str
)

"""
`Observation`:
* Only GPT4v and GogVLM output desired fields
* Among those 2 models, GPT4V gets more accurate results.
"""

logger.info("\n\n[DONE]", bright=True)