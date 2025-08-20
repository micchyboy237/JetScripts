from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.modelscope import ModelScopeLLM
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/modelscope.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ModelScope LLMS

In this notebook, we show how to use the ModelScope LLM models in LlamaIndex. Check out the [ModelScope site](https://www.modelscope.cn/).

If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the modelscope.
"""
logger.info("# ModelScope LLMS")

# !pip install llama-index-llms-modelscope

"""
## Basic Usage
"""
logger.info("## Basic Usage")


llm = ModelScopeLLM(model_name="qwen/Qwen3-8B", model_revision="master")

rsp = llm.complete("Hello, who are you?")
logger.debug(rsp)

"""
#### Use Message request
"""
logger.info("#### Use Message request")


messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content="How to make cake?"),
]
resp = llm.chat(messages)
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)