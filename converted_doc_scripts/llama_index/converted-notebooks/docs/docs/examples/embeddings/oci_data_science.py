import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.oci_data_science import OCIDataScienceEmbedding
import ads
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/oci_data_science.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Oracle Cloud Infrastructure (OCI) Data Science Service

Oracle Cloud Infrastructure (OCI) [Data Science](https://www.oracle.com/artificial-intelligence/data-science) is a fully managed, serverless platform for data science teams to build, train, and manage machine learning models in Oracle Cloud Infrastructure.

It offers [AI Quick Actions](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm), which can be used to deploy embedding models in OCI Data Science. AI Quick Actions target users who want to quickly leverage the capabilities of AI. They aim to expand the reach of foundation models to a broader set of users by providing a streamlined, code-free, and efficient environment for working with foundation models. AI Quick Actions can be accessed from the Data Science Notebook.

Detailed documentation on how to deploy embedding models in OCI Data Science using AI Quick Actions is available [here](https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md) and [here](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm).

This notebook explains how to use OCI's Data Science embedding models with LlamaIndex.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Oracle Cloud Infrastructure (OCI) Data Science Service")

# %pip install llama-index-embeddings-oci-data-science

# !pip install llama-index

"""
You will also need to install the [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) SDK.
"""
logger.info("You will also need to install the [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) SDK.")

# !pip install -U oracle-ads

"""
## Authentication

The authentication methods supported for LlamaIndex are equivalent to those used with other OCI services and follow the standard SDK authentication methods, specifically API Key, session token, instance principal, and resource principal. More details can be found [here](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html). Make sure to have the required [policies](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm) to access the OCI Data Science Model Deployment endpoint. The [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) helps to simplify the authentication within OCI Data Science.

## Basic Usage
"""
logger.info("## Authentication")


ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

embedding = OCIDataScienceEmbedding(
    endpoint="https://<MD_OCID>/predict",
)


e1 = embeddings.get_text_embedding("This is a test document")
logger.debug(e1)

e2 = embeddings.get_text_embedding_batch(
    ["This is a test document", "This is another test document"]
)
logger.debug(e2)

"""
## Async
"""
logger.info("## Async")


ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

embedding = OCIDataScienceEmbedding(
    endpoint="https://<MD_OCID>/predict",
)

async def run_async_code_0473680b():
    async def run_async_code_5a4152ad():
        e1 = await embeddings.aget_text_embedding("This is a test document")
        return e1
    e1 = asyncio.run(run_async_code_5a4152ad())
    logger.success(format_json(e1))
    return e1
e1 = asyncio.run(run_async_code_0473680b())
logger.success(format_json(e1))
logger.debug(e1)

async def async_func_12():
    e2 = await embeddings.aget_text_embedding_batch(
        ["This is a test document", "This is another test document"]
    )
    return e2
e2 = asyncio.run(async_func_12())
logger.success(format_json(e2))
logger.debug(e2)

logger.info("\n\n[DONE]", bright=True)