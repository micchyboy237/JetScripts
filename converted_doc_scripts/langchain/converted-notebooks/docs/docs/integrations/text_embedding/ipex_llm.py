from jet.logger import logger
from langchain_community.embeddings import IpexLLMBgeEmbeddings
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
# IPEX-LLM: Local BGE Embeddings on Intel CPU

> [IPEX-LLM](https://github.com/intel-analytics/ipex-llm) is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency.

This example goes over how to use LangChain to conduct embedding tasks with `ipex-llm` optimizations on Intel CPU. This would be helpful in applications such as RAG, document QA, etc.

## Setup
"""
logger.info("# IPEX-LLM: Local BGE Embeddings on Intel CPU")

# %pip install -qU langchain langchain-community

"""
Install IPEX-LLM for optimizations on Intel CPU, as well as `sentence-transformers`.
"""
logger.info("Install IPEX-LLM for optimizations on Intel CPU, as well as `sentence-transformers`.")

# %pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
# %pip install sentence-transformers

"""
> **Note**
>
> For Windows users, `--extra-index-url https://download.pytorch.org/whl/cpu` when install `ipex-llm` is not required.

## Basic Usage
"""
logger.info("## Basic Usage")


embedding_model = IpexLLMBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={},
    encode_kwargs={"normalize_embeddings": True},
)

"""
API Reference
- [IpexLLMBgeEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.ipex_llm.IpexLLMBgeEmbeddings.html)
"""
logger.info("API Reference")

sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"

text_embeddings = embedding_model.embed_documents([sentence, query])
logger.debug(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
logger.debug(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

query_embedding = embedding_model.embed_query(query)
logger.debug(f"query_embedding[:10]: {query_embedding[:10]}")

logger.info("\n\n[DONE]", bright=True)