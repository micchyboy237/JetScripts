from jet.logger import logger
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import (
GemmaChatVertexAIModelGarden,
GemmaVertexAIModelGarden,
)
from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF
from langchain_google_vertexai import GemmaChatLocalKaggle
from langchain_google_vertexai import GemmaLocalKaggle
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
## Getting started with LangChain and Gemma, running locally or in the Cloud

### Installing dependencies
"""
logger.info("## Getting started with LangChain and Gemma, running locally or in the Cloud")

# !pip install --upgrade langchain langchain-google-vertexai

"""
### Running the model

Go to the VertexAI Model Garden on Google Cloud [console](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/335), and deploy the desired version of Gemma to VertexAI. It will take a few minutes, and after the endpoint is ready, you need to copy its number.
"""
logger.info("### Running the model")

project: str = "PUT_YOUR_PROJECT_ID_HERE"  # @param {type:"string"}
endpoint_id: str = "PUT_YOUR_ENDPOINT_ID_HERE"  # @param {type:"string"}
location: str = "PUT_YOUR_ENDPOINT_LOCAtION_HERE"  # @param {type:"string"}


llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

output = llm.invoke("What is the meaning of life?")
logger.debug(output)

"""
We can also use Gemma as a multi-turn chat model:
"""
logger.info("We can also use Gemma as a multi-turn chat model:")


llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
logger.debug(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

logger.debug(answer2)

"""
You can post-process response to avoid repetitions:
"""
logger.info("You can post-process response to avoid repetitions:")

answer1 = llm.invoke([message1], parse_response=True)
logger.debug(answer1)

answer2 = llm.invoke([message1, answer1, message2], parse_response=True)

logger.debug(answer2)

"""
## Running Gemma locally from Kaggle

In order to run Gemma locally, you can download it from Kaggle first. In order to do this, you'll need to login into the Kaggle platform, create a API key and download a `kaggle.json` Read more about Kaggle auth [here](https://www.kaggle.com/docs/api).

### Installation
"""
logger.info("## Running Gemma locally from Kaggle")

# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json

# !pip install keras>=3 keras_nlp

"""
### Usage
"""
logger.info("### Usage")


"""
You can specify the keras backend (by default it's `tensorflow`, but you can change it be `jax` or `torch`).
"""
logger.info("You can specify the keras backend (by default it's `tensorflow`, but you can change it be `jax` or `torch`).")

keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}

llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)

output = llm.invoke("What is the meaning of life?", max_tokens=30)
logger.debug(output)

"""
### ChatModel

Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:
"""
logger.info("### ChatModel")


keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}

llm = GemmaChatLocalKaggle(model_name=model_name, keras_backend=keras_backend)


message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=30)
logger.debug(answer1)

message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=60)

logger.debug(answer2)

"""
You can post-process the response if you want to avoid multi-turn statements:
"""
logger.info("You can post-process the response if you want to avoid multi-turn statements:")

answer1 = llm.invoke([message1], max_tokens=30, parse_response=True)
logger.debug(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=60, parse_response=True)
logger.debug(answer2)

"""
## Running Gemma locally from HuggingFace
"""
logger.info("## Running Gemma locally from HuggingFace")


hf_access_token: str = "PUT_YOUR_TOKEN_HERE"  # @param {type:"string"}
model_name: str = "google/gemma-2b"  # @param {type:"string"}

llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)

output = llm.invoke("What is the meaning of life?", max_tokens=50)
logger.debug(output)

"""
Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:
"""
logger.info("Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:")

llm = GemmaChatLocalHF(model_name=model_name, hf_access_token=hf_access_token)


message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=60)
logger.debug(answer1)

message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=140)

logger.debug(answer2)

"""
And the same with posprocessing:
"""
logger.info("And the same with posprocessing:")

answer1 = llm.invoke([message1], max_tokens=60, parse_response=True)
logger.debug(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=120, parse_response=True)
logger.debug(answer2)

logger.info("\n\n[DONE]", bright=True)