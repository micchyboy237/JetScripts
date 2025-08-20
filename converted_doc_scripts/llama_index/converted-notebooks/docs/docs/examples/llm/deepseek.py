from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/deepseek.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DeepSeek

# LlamaIndex Llms Integration: DeepSeek

This is the DeepSeek integration for LlamaIndex. Visit [DeepSeek](https://api-docs.deepseek.com/) for information on how to get an API key and which models are supported. 

At the time of writing, you can use:
- `deepseek-chat`
- `deepseek-reasoner`

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# DeepSeek")

# %pip install llama-index-llms-deepseek


llm = DeepSeek(model="deepseek-reasoner", api_key="you_api_key")

response = llm.complete("Is 9.9 or 9.11 bigger?")

logger.debug(response)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(
        role="user", content="How many 'r's are in the word 'strawberry'?"
    ),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Is 9.9 or 9.11 bigger?")

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
    ChatMessage(
        role="user", content="How many 'r's are in the word 'strawberry'?"
    ),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)