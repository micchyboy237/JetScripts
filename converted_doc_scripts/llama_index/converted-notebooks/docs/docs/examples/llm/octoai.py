from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.octoai import OctoAI
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/octoai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OctoAI

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OctoAI")

# %pip install llama-index-llms-octoai
# %pip install llama-index
# %pip install octoai-sdk

"""
Include your OctoAI API key below. You can get yours at [OctoAI](https://octo.ai). 

[Here](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token) are some instructions in case you need more guidance.
"""
logger.info("Include your OctoAI API key below. You can get yours at [OctoAI](https://octo.ai).")

OCTOAI_API_KEY = ""

"""
#### Initialize the Integration with the default model
"""
logger.info("#### Initialize the Integration with the default model")


octoai = OctoAI(token=OCTOAI_API_KEY)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")

response = octoai.complete("Paul Graham is ")
logger.debug(response)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system",
        content="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    ),
    ChatMessage(role="user", content="Write a blog about Seattle"),
]
response = octoai.chat(messages)
logger.debug(response)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

response = octoai.stream_complete("Paul Graham is ")
for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` with a list of messages
"""
logger.info("Using `stream_chat` with a list of messages")


messages = [
    ChatMessage(
        role="system",
        content="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    ),
    ChatMessage(role="user", content="Write a blog about Seattle"),
]
response = octoai.stream_chat(messages)
for r in response:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")

octoai = OctoAI(
    model="mistral-7b-instruct", max_tokens=128, token=OCTOAI_API_KEY
)

response = octoai.complete("Paul Graham is ")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)