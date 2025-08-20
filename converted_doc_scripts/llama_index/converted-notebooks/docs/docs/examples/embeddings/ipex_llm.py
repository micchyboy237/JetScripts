from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
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
# Local Embeddings with IPEX-LLM on Intel CPU

> [IPEX-LLM](https://github.com/intel-analytics/ipex-llm/) is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency.

This example goes over how to use LlamaIndex to conduct embedding tasks with `ipex-llm` optimizations on Intel CPU. This would be helpful in applications such as RAG, document QA, etc.

> **Note**
>
> You could refer to [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-ipex-llm/examples) for full examples of `IpexLLMEmbedding`. Please note that for running on Intel CPU, please specify `-d 'cpu'` in command argument when running the examples.

## Install `llama-index-embeddings-ipex-llm`

This will also install `ipex-llm` and its dependencies.
"""
logger.info("# Local Embeddings with IPEX-LLM on Intel CPU")

# %pip install llama-index-embeddings-ipex-llm

"""
## `IpexLLMEmbedding`
"""
logger.info("## `IpexLLMEmbedding`")


embedding_model = IpexLLMEmbedding(model_name="BAAI/bge-large-en-v1.5")

"""
> Please note that `IpexLLMEmbedding` currently only provides optimization for Hugging Face Bge models.
"""

sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"

text_embedding = embedding_model.get_text_embedding(sentence)
logger.debug(f"embedding[:10]: {text_embedding[:10]}")

text_embeddings = embedding_model.get_text_embedding_batch([sentence, query])
logger.debug(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
logger.debug(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

query_embedding = embedding_model.get_query_embedding(query)
logger.debug(f"query_embedding[:10]: {query_embedding[:10]}")

logger.info("\n\n[DONE]", bright=True)