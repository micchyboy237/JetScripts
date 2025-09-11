from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Hugging Face
Let's load the Hugging Face Embedding class.
"""
logger.info("# Hugging Face")

# %pip install --upgrade --quiet  langchain langchain-huggingface sentence_transformers


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

text = "This is a test document."

query_result = embeddings.embed_query(text)

query_result[:3]

doc_result = embeddings.embed_documents([text])

"""
## Hugging Face Inference Providers

We can also access embedding models via the [Inference Providers](https://huggingface.co/docs/inference-providers), which let's us use open source models on scalable serverless infrastructure.

First, we need to get a read-only API key from [Hugging Face](https://huggingface.co/settings/tokens).
"""
logger.info("## Hugging Face Inference Providers")

# from getpass import getpass

# huggingfacehub_api_token = getpass()

"""
Now we can use the `HuggingFaceInferenceAPIEmbeddings` class to run open source embedding models via [Inference Providers](https://huggingface.co/docs/inference-providers).
"""
logger.info("Now we can use the `HuggingFaceInferenceAPIEmbeddings` class to run open source embedding models via [Inference Providers](https://huggingface.co/docs/inference-providers).")


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=huggingfacehub_api_token,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)

query_result = embeddings.embed_query(text)
query_result[:3]

"""
## Hugging Face Hub
We can also generate embeddings locally via the Hugging Face Hub package, which requires us to install ``huggingface_hub ``
"""
logger.info("## Hugging Face Hub")

# !pip install huggingface_hub


embeddings = HuggingFaceEndpointEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)

query_result[:3]

logger.info("\n\n[DONE]", bright=True)