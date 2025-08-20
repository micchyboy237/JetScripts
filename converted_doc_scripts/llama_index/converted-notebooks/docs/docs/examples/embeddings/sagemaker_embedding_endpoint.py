from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.sagemaker_endpoint import SageMakerEmbedding
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

# Interacting with Embeddings deployed in Amazon SageMaker Endpoint with LlamaIndex

An Amazon SageMaker endpoint is a fully managed resource that enables the deployment of machine learning models, for making predictions on new data.

This notebook demonstrates how to interact with Embedding endpoints using `SageMakerEmbedding`, unlocking additional llamaIndex features.
So, It is assumed that an Embedding is deployed on a SageMaker endpoint.

## Setting Up
If you’re opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Interacting with Embeddings deployed in Amazon SageMaker Endpoint with LlamaIndex")

# %pip install llama-index-embeddings-sagemaker-endpoint

# ! pip install llama-index

"""
You have to specify the endpoint name to interact with.
"""
logger.info("You have to specify the endpoint name to interact with.")

ENDPOINT_NAME = "<-YOUR-ENDPOINT-NAME->"

"""
Credentials should be provided to connect to the endpoint. You can either:
-  use an AWS profile by specifying the `profile_name` parameter, if not specified, the default credential profile will be used. 
-  Pass credentials as parameters (`aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `region_name`). 

for more details check [this link](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

**AWS profile name**
"""
logger.info("Credentials should be provided to connect to the endpoint. You can either:")


AWS_ACCESS_KEY_ID = "<-YOUR-AWS-ACCESS-KEY-ID->"
AWS_SECRET_ACCESS_KEY = "<-YOUR-AWS-SECRET-ACCESS-KEY->"
AWS_SESSION_TOKEN = "<-YOUR-AWS-SESSION-TOKEN->"
REGION_NAME = "<-YOUR-ENDPOINT-REGION-NAME->"

embed_model = SageMakerEmbedding(
    endpoint_name=ENDPOINT_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=REGION_NAME,
)

"""
**With credentials**:
"""


ENDPOINT_NAME = "<-YOUR-ENDPOINT-NAME->"
PROFILE_NAME = "<-YOUR-PROFILE-NAME->"
embed_model = SageMakerEmbedding(
    endpoint_name=ENDPOINT_NAME, profile_name=PROFILE_NAME
)  # Omit the profile name to use the default profile

"""
## Basic Usage

### Call `get_text_embedding`
"""
logger.info("## Basic Usage")

embeddings = embed_model.get_text_embedding(
    "An Amazon SageMaker endpoint is a fully managed resource that enables the deployment of machine learning models, specifically LLM (Large Language Models), for making predictions on new data."
)

embeddings

"""
### Call `get_text_embedding_batch`
"""
logger.info("### Call `get_text_embedding_batch`")

embeddings = embed_model.get_text_embedding_batch(
    [
        "An Amazon SageMaker endpoint is a fully managed resource that enables the deployment of machine learning models",
        "Sagemaker is integrated with llamaIndex",
    ]
)

len(embeddings)

logger.info("\n\n[DONE]", bright=True)