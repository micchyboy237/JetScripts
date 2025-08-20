from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.databricks import Databricks
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
# Databricks

Integrate with Databricks LLMs APIs.

## Pre-requisites

- [Databricks personal access token](https://docs.databricks.com/en/dev-tools/auth/pat.html) to query and access Databricks model serving endpoints.

- [Databricks workspace](https://docs.databricks.com/en/workspace/index.html) in a [supported region](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#regions) for Foundation Model APIs pay-per-token.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Databricks")

# % pip install llama-index-llms-databricks

# !pip install llama-index


"""
```bash
export DATABRICKS_TOKEN=<your api key>
export DATABRICKS_SERVING_ENDPOINT=<your api serving endpoint>
```

Alternatively, you can pass your API key and serving endpoint to the LLM when you init it:
"""
logger.info("export DATABRICKS_TOKEN=<your api key>")

llm = Databricks(
    model="databricks-dbrx-instruct",
    api_key="your_api_key",
    api_base="https://[your-work-space].cloud.databricks.com/serving-endpoints/",
)

"""
A list of available LLM models can be found [here](https://console.groq.com/docs/models).
"""
logger.info("A list of available LLM models can be found [here](https://console.groq.com/docs/models).")

response = llm.complete("Explain the importance of open source LLMs")

logger.debug(response)

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
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Explain the importance of open source LLMs")

for r in response:
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

logger.info("\n\n[DONE]", bright=True)