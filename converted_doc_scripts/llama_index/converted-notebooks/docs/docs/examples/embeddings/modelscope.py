from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.modelscope.base import ModelScopeEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/modelscope.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ModelScope Embeddings

In this notebook, we show how to use the ModelScope Embeddings in LlamaIndex. Check out the [ModelScope site](https://www.modelscope.cn/).

If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the modelscope.
"""
logger.info("# ModelScope Embeddings")

# !pip install llama-index-embeddings-modelscope

"""
## Basic Usage
"""
logger.info("## Basic Usage")


model = ModelScopeEmbedding(
    model_name="iic/nlp_gte_sentence-embedding_chinese-base",
    model_revision="master",
)

rsp = model.get_query_embedding("Hello, who are you?")
logger.debug(rsp)

rsp = model.get_text_embedding("Hello, who are you?")
logger.debug(rsp)

"""
#### Generate Batch Embedding
"""
logger.info("#### Generate Batch Embedding")


model = ModelScopeEmbedding(
    model_name="iic/nlp_gte_sentence-embedding_chinese-base",
    model_revision="master",
)

rsp = model.get_text_embedding_batch(
    ["Hello, who are you?", "I am a student."]
)
logger.debug(rsp)

logger.info("\n\n[DONE]", bright=True)