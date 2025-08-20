from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.vertex_endpoint import VertexEndpointEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/sagemaker_embedding_endpoint.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Interacting with Embeddings deployed in Vertex AI Endpoint with LlamaIndex

A Vertex AI endpoint is a managed resource that enables the deployment of machine learning models, such as embeddings, for making predictions on new data.

This notebook demonstrates how to interact with embedding endpoints using the `VertexEndpointEmbedding` class, leveraging the LlamaIndex.

## Setting Up
If you’re opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Interacting with Embeddings deployed in Vertex AI Endpoint with LlamaIndex")

# %pip install llama-index-embeddings-vertex-endpoint

# ! pip install llama-index

"""
You need to specify the endpoint information (endpoint ID, project ID, and region) to interact with the model deployed in Vertex AI.
"""
logger.info("You need to specify the endpoint information (endpoint ID, project ID, and region) to interact with the model deployed in Vertex AI.")

ENDPOINT_ID = "<-YOUR-ENDPOINT-ID->"
PROJECT_ID = "<-YOUR-PROJECT-ID->"
LOCATION = "<-YOUR-GCP-REGION->"

"""
Credentials should be provided to connect to the endpoint. You can either:

- Use a service account JSON file by specifying the `service_account_file` parameter.
- Provide the service account information directly through the `service_account_info` parameter.

**Example using a service account file:**
"""
logger.info("Credentials should be provided to connect to the endpoint. You can either:")


SERVICE_ACCOUNT_FILE = "<-YOUR-SERVICE-ACCOUNT-FILE-PATH->.json"

embed_model = VertexEndpointEmbedding(
    endpoint_id=ENDPOINT_ID,
    project_id=PROJECT_ID,
    location=LOCATION,
    service_account_file=SERVICE_ACCOUNT_FILE,
)

"""
**Example using direct service account info:**:
"""


SERVICE_ACCOUNT_INFO = {
    "private_key": "<-PRIVATE-KEY->",
    "client_email": "<-SERVICE-ACCOUNT-EMAIL->",
    "token_uri": "https://oauth2.googleapis.com/token",
}

embed_model = VertexEndpointEmbedding(
    endpoint_id=ENDPOINT_ID,
    project_id=PROJECT_ID,
    location=LOCATION,
    service_account_info=SERVICE_ACCOUNT_INFO,
)

"""
## Basic Usage

### Call `get_text_embedding`
"""
logger.info("## Basic Usage")

embeddings = embed_model.get_text_embedding(
    "Vertex AI is a managed machine learning (ML) platform provided by Google Cloud. It allows data scientists and developers to build, deploy, and scale machine learning models efficiently, leveraging Google's ML infrastructure."
)

embeddings[:10]

"""
### Call `get_text_embedding_batch`
"""
logger.info("### Call `get_text_embedding_batch`")

embeddings = embed_model.get_text_embedding_batch(
    [
        "Vertex AI is a managed machine learning (ML) platform provided by Google Cloud. It allows data scientists and developers to build, deploy, and scale machine learning models efficiently, leveraging Google's ML infrastructure.",
        "Vertex is integrated with llamaIndex",
    ]
)

len(embeddings)

logger.info("\n\n[DONE]", bright=True)