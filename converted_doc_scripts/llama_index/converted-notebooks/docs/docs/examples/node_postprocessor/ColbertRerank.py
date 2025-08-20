from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/ColbertRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Colbert Rerank

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.


[Colbert](https://github.com/stanford-futuredata/ColBERT): ColBERT is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds.

This example shows how we use Colbert-V2 model as a reranker.
"""
logger.info("# Colbert Rerank")

# !pip install llama-index
# !pip install llama-index-core
# !pip install --quiet transformers torch
# !pip install llama-index-embeddings-ollama
# !pip install llama-index-llms-ollama
# !pip install llama-index-postprocessor-colbert-rerank


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


# os.environ["OPENAI_API_KEY"] = "sk-"

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents=documents)

"""
#### Retrieve top 10 most relevant nodes, then filter with Colbert Rerank
"""
logger.info("#### Retrieve top 10 most relevant nodes, then filter with Colbert Rerank")


colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[colbert_reranker],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

for node in response.source_nodes:
    logger.debug(node.id_)
    logger.debug(node.node.get_content()[:120])
    logger.debug("reranking score: ", node.score)
    logger.debug("retrieval score: ", node.node.metadata["retrieval_score"])
    logger.debug("**********")

logger.debug(response)

response = query_engine.query(
    "Which schools did Paul attend?",
)

for node in response.source_nodes:
    logger.debug(node.id_)
    logger.debug(node.node.get_content()[:120])
    logger.debug("reranking score: ", node.score)
    logger.debug("retrieval score: ", node.node.metadata["retrieval_score"])
    logger.debug("**********")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)