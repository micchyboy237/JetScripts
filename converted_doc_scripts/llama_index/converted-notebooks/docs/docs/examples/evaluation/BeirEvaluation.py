from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/BeirEvaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# BEIR Out of Domain Benchmark

About [BEIR](https://github.com/beir-cellar/beir):

BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your retrieval methods within the benchmark.

Refer to the repo via the link for a full list of supported datasets.

Here, we test the `all-MiniLM-L6-v2` sentence-transformer embedding, which is one of the fastest for the given accuracy range. We set the top_k value for the retriever to 30. We also use the nfcorpus dataset.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# BEIR Out of Domain Benchmark")

# %pip install llama-index-embeddings-huggingface

# !pip install llama-index



def create_retriever(documents):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, show_progress=True
    )
    return index.as_retriever(similarity_top_k=30)


BeirEvaluator().run(
    create_retriever, datasets=["nfcorpus"], metrics_k_values=[3, 10, 30]
)

"""
Higher is better for all the evaluation metrics.

This [towardsdatascience article](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54) covers NDCG, MAP and MRR in greater depth.
"""
logger.info("Higher is better for all the evaluation metrics.")

logger.info("\n\n[DONE]", bright=True)