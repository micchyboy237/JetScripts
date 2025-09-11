from jet.logger import logger
from langchain_oci.embeddings import OCIGenAIEmbeddings
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
# Oracle Cloud Infrastructure Generative AI

Oracle Cloud Infrastructure (OCI) Generative AI is a fully managed service that provides a set of state-of-the-art, customizable large language models (LLMs), that cover a wide range of use cases, and which are available through a single API.
Using the OCI Generative AI service you can access ready-to-use pretrained models, or create and host your own fine-tuned custom models based on your own data on dedicated AI clusters. Detailed documentation of the service and API is available __[here](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)__ and __[here](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/)__.

This notebook explains how to use OCI's Genrative AI models with LangChain.

### Prerequisite
We will need to install the oci sdk
"""
logger.info("# Oracle Cloud Infrastructure Generative AI")

# !pip install -U langchain_oci

"""
### OCI Generative AI API endpoint 
https://inference.generativeai.us-chicago-1.oci.oraclecloud.com

## Authentication
The authentication methods supported for this langchain integration are:

1. API Key
2. Session token
3. Instance principal
4. Resource principal 

These follows the standard SDK authentication methods detailed __[here](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm)__.

## Usage
"""
logger.info("### OCI Generative AI API endpoint")


embeddings = OCIGenAIEmbeddings(
    model_id="MY_EMBEDDING_MODEL",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_OCID",
)


query = "This is a query in English."
response = embeddings.embed_query(query)
logger.debug(response)

documents = ["This is a sample document", "and here is another one"]
response = embeddings.embed_documents(documents)
logger.debug(response)

embeddings = OCIGenAIEmbeddings(
    model_id="MY_EMBEDDING_MODEL",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_OCID",
    auth_type="SECURITY_TOKEN",
    auth_profile="MY_PROFILE",  # replace with your profile name
    auth_file_location="MY_CONFIG_FILE_LOCATION",  # replace with file location where profile name configs present
)


query = "This is a sample query"
response = embeddings.embed_query(query)
logger.debug(response)

documents = ["This is a sample document", "and here is another one"]
response = embeddings.embed_documents(documents)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)