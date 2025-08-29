from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.baseten import Baseten
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Baseten Cookbook
"""
logger.info("# Baseten Cookbook")

# %pip install llama-index llama-index-llms-baseten


"""
## Model APIs vs. Dedicated Deployments

Baseten offers two main ways for inference.
1. Model APIs are public endpoints for popular open source models (GPT-OSS, Kimi K2, DeepSeek etc) where you can directly use a frontier model via slug e.g.  `deepseek-ai/DeepSeek-V3-0324` and you will be charged on a per-token basis. You can find the list of supported models here: https://docs.baseten.co/development/model-apis/overview#supported-models.

2. Dedicated deployments are useful for serving custom models where you want to autoscale production workloads and have fine-grain configuration. You need to deploy a model in your Baseten dashboard and provide the 8 character model id like `abcd1234`.

By default, we set the `model_apis` parameter to `True`. If you want to use a dedicated deployment, you must set the `model_apis` parameter to `False` when instantiating the Baseten object.

#### Instantiation
"""
logger.info("## Model APIs vs. Dedicated Deployments")

llm = Baseten(
    model_id="MODEL_SLUG",
    api_key="YOUR_API_KEY",
    model_apis=True,  # Default, so not strictly necessary
)

llm = Baseten(
    model_id="MODEL_ID",
    api_key="YOUR_API_KEY",
    model_apis=False,
)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")

llm_response = llm.complete("Paul Graham is")
logger.debug(llm_response.text)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


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
# Async
Async operations are used for long-running inference tasks that may hit request timeouts, batch inference jobs, and prioritizing certain requests.

(1) In the integation, `acomplete` async function is implemented using the aiohttp library, an asynchronous HTTP client in python. The function invokes the async_predict at the approriate Baseten model endpoint, then the user receives a response with the request_id if successful. The user can then check the status or cancel the async_predict request using the returned request_id.

(2) Once the model finishes executing the request, the async result will be posted to the user provided webhook endpoint. The user's endpoint is responsible for validating the webhook signature for security, then processing and storing the output.

Baseten: Get request_id → result is posted to webhook

##### Note: Async is only available for dedicated deployments and not for model APIs. `achat` is not supported because chat does not make sense for async operations.
"""
logger.info("# Async")

async_llm = Baseten(
    model_id="YOUR_MODEL_ID",
    api_key="YOUR_API_KEY",
    webhook_endpoint="YOUR_WEBHOOK_ENDPOINT",
)
response = async_llm.complete("Paul Graham is")
logger.success(format_json(response))
logger.success(format_json(response))
logger.debug(response)  # This is the request id

"""
This will return the status information of a request using an async_predict request's request_id and the model_id the async_predict request was made with.
"""


model_id = "YOUR_MODEL_ID"
request_id = "YOUR_REQUEST_ID"
baseten_api_key = "YOUR_API_KEY"

resp = requests.get(
    f"https://model-{model_id}.api.baseten.co/async_request/{request_id}",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
)

logger.debug(resp.json())

logger.info("\n\n[DONE]", bright=True)