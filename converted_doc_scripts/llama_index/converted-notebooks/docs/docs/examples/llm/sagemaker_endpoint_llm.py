from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
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
<a href="https://colab.research.google.com/drive/104BZb4U1KLzOYArnCzhqN-CqJWZDeqhz?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Interacting with LLM deployed in Amazon SageMaker Endpoint with LlamaIndex

An Amazon SageMaker endpoint is a fully managed resource that enables the deployment of machine learning models, specifically LLM (Large Language Models), for making predictions on new data.

This notebook demonstrates how to interact with LLM endpoints using `SageMakerLLM`, unlocking additional llamaIndex features.
So, It is assumed that an LLM is deployed on a SageMaker endpoint.

## Setting Up
If you’re opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Interacting with LLM deployed in Amazon SageMaker Endpoint with LlamaIndex")

# %pip install llama-index-llms-sagemaker-endpoint

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

llm = SageMakerLLM(
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
llm = SageMakerLLM(
    endpoint_name=ENDPOINT_NAME, profile_name=PROFILE_NAME
)  # Omit the profile name to use the default profile

"""
## Basic Usage

### Call `complete` with a prompt
"""
logger.info("## Basic Usage")

resp = llm.complete(
    "Paul Graham is ", formatted=True
)  # formatted=True to avoid adding system prompt
logger.debug(resp)

"""
### Call `chat` with a list of messages
"""
logger.info("### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
### Streaming

#### Using `stream_complete` endpoint
"""
logger.info("### Streaming")

resp = llm.stream_complete("Paul Graham is ", formatted=True)

for r in resp:
    logger.debug(r.delta)

"""
#### Using `stream_chat` endpoint
"""
logger.info("#### Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model

`SageMakerLLM` is an abstraction for interacting with different language models (LLM) deployed in Amazon SageMaker. All the default parameters are compatible with the Llama 2 model. Therefore, if you are using a different model, you will likely need to set the following parameters:

- `messages_to_prompt`: A callable that accepts a list of `ChatMessage` objects and, if not specified in the message, a system prompt. It should return a string containing the messages in the endpoint LLM-compatible format.

- `completion_to_prompt`: A callable that accepts a completion string with a system prompt and returns a string in the endpoint LLM-compatible format.

- `content_handler`: A class that inherits from `llama_index.llms.sagemaker_llm_endpoint_utils.BaseIOHandler` and implements the following methods: `serialize_input`, `deserialize_output`, `deserialize_streaming_output`, and `remove_prefix`.
"""
logger.info("## Configure Model")

logger.info("\n\n[DONE]", bright=True)