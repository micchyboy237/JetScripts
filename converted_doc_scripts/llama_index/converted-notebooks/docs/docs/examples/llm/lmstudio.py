from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/lmstudio.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LM Studio

## Setup

1. Download and Install LM Studio
2. Follow the steps mentioned in the [README](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-lmstudio/README.md).

If not already installed in collab, install *llama-index* and *lmstudio* integration.
"""
logger.info("# LM Studio")

# %pip install llama-index-core llama-index llama-index-llms-lmstudio

"""
Fix for "RuntimeError: This event loop is already running"
"""
logger.info("Fix for "RuntimeError: This event loop is already running"")

# import nest_asyncio

# nest_asyncio.apply()


llm = LMStudio(
    model_name="Hermes-2-Pro-Llama-3-8B",
    base_url="http://localhost:1234/v1",
    temperature=0.7,
)

response = llm.complete("Hey there, what is 2+2?")
logger.debug(str(response))

response = llm.stream_complete("What is 7+3?")
for r in response:
    logger.debug(r.delta, end="")

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You an expert AI assistant. Help User with their queries.",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What is the significance of the number 42?",
    ),
]

response = llm.chat(messages=messages)
logger.debug(str(response))

response = llm.stream_chat(messages=messages)
for r in response:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)