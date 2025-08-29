from jet.logger import CustomLogger
from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/bedrock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Oracle Cloud Infrastructure Generative AI

Oracle Cloud Infrastructure (OCI) Generative AI is a fully managed service that provides a set of state-of-the-art, customizable large language models (LLMs) that cover a wide range of use cases, and which is available through a single API.
Using the OCI Generative AI service you can access ready-to-use pretrained models, or create and host your own fine-tuned custom models based on your own data on dedicated AI clusters. Detailed documentation of the service and API is available __[here](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)__ and __[here](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/)__.

This notebook explains how to use OCI's Genrative AI embedding models with LlamaIndex.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Oracle Cloud Infrastructure Generative AI")

# %pip install llama-index-embeddings-oci-genai

# !pip install llama-index

"""
You will also need to install the OCI sdk
"""
logger.info("You will also need to install the OCI sdk")

# !pip install -U oci

"""
## Basic Usage
"""
logger.info("## Basic Usage")


embedding = OCIGenAIEmbeddings(
    model_name="cohere.embed-english-light-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_OCID",
)

e1 = embedding.get_text_embedding("This is a test document")
logger.debug(e1[-5:])

e2 = embedding.get_query_embedding("This is a test document")
logger.debug(e2[-5:])

docs = ["This is a test document", "This is another test document"]
e3 = embedding.get_text_embedding_batch(docs)
logger.debug(e3)

logger.info("\n\n[DONE]", bright=True)