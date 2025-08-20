from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
import os
import requests
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/JinaRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Jina Rerank

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Jina Rerank")

# !pip install llama-index-postprocessor-jinaai-rerank
# !pip install llama-index-embeddings-jinaai
# !pip install llama-index



api_key = os.environ["JINA_API_KEY"]
jina_embeddings = JinaEmbedding(api_key=api_key)


url = "https://niketeam-asset-download.nike.net/catalogs/2024/2024_Nike%20Kids_02_09_24.pdf?cb=09302022"
response = requests.get(url)
with open("Nike_Catalog.pdf", "wb") as f:
    f.write(response.content)
reader = SimpleDirectoryReader(input_files=["Nike_Catalog.pdf"])
documents = reader.load_data()

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=jina_embeddings
)

"""
#### Retrieve top 10 most relevant nodes, without using a reranker
"""
logger.info("#### Retrieve top 10 most relevant nodes, without using a reranker")

query_engine = index.as_query_engine(similarity_top_k=10)
response = query_engine.query(
    "What is the best jersey by Nike in terms of fabric?",
)

logger.debug(response.source_nodes[0].text, response.source_nodes[0].score)
logger.debug("\n")
logger.debug(response.source_nodes[1].text, response.source_nodes[1].score)

"""
#### Retrieve top 10 most relevant nodes, but then rerank using Jina Reranker

By employing a reranker model, the prompt can be given more relevant context. This will lead to a more accurate response by the LLM.
"""
logger.info("#### Retrieve top 10 most relevant nodes, but then rerank using Jina Reranker")


jina_rerank = JinaRerank(api_key=api_key, top_n=2)

query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[jina_rerank]
)
response = query_engine.query(
    "What is the best jersey by Nike in terms of fabric?",
)

logger.debug(response.source_nodes[0].text, response.source_nodes[0].score)
logger.debug("\n")
logger.debug(response.source_nodes[1].text, response.source_nodes[1].score)

logger.info("\n\n[DONE]", bright=True)