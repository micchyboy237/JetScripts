from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
import openai
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/relative_score_dist_fusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Relative Score Fusion and Distribution-Based Score Fusion

In this example, we demonstrate using QueryFusionRetriever with two methods which aim to improve on Reciprocal Rank Fusion:
1. Relative Score Fusion ([Weaviate](https://weaviate.io/blog/hybrid-search-fusion-algorithms))
2. Distribution-Based Score Fusion ([Mazzeschi: blog post](https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18))
"""
logger.info("# Relative Score Fusion and Distribution-Based Score Fusion")

# %pip install llama-index-llms-ollama
# %pip install llama-index-retrievers-bm25


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

Download Data
"""
logger.info("## Setup")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
Next, we will setup a vector index over the documentation.
"""
logger.info("Next, we will setup a vector index over the documentation.")


splitter = SentenceSplitter(chunk_size=256)

index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], show_progress=True
)

"""
## Create a Hybrid Fusion Retriever using Relative Score Fusion

In this step, we fuse our index with a BM25 based retriever. This will enable us to capture both semantic relations and keywords in our input queries.

Since both of these retrievers calculate a score, we can use the `QueryFusionRetriever` to re-sort our nodes without using an additional models or excessive computation.

The following example uses the [Relative Score Fusion](https://weaviate.io/blog/hybrid-search-fusion-algorithms) algorithm from Weaviate, which applies a MinMax scaler to each result set, then makes a weighted sum. Here, we'll give the vector retriever slightly more weight than BM25 (0.6 vs. 0.4).

First, we create our retrievers. Each will retrieve the top-10 most similar nodes.
"""
logger.info("## Create a Hybrid Fusion Retriever using Relative Score Fusion")


vector_retriever = index.as_retriever(similarity_top_k=5)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=10
)

"""
Next, we can create our fusion retriever, which well return the top-10 most similar nodes from the 20 returned nodes from the retrievers.

Note that the vector and BM25 retrievers may have returned all the same nodes, only in different orders; in this case, it simply acts as a re-ranker.
"""
logger.info("Next, we can create our fusion retriever, which well return the top-10 most similar nodes from the 20 returned nodes from the retrievers.")


retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="relative_score",
    use_async=True,
    verbose=True,
)

# import nest_asyncio

# nest_asyncio.apply()

nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

for node in nodes_with_scores:
    logger.debug(f"Score: {node.score:.2f} - {node.text[:100]}...\n-----")

"""
### Distribution-Based Score Fusion

A variant on Relative Score Fusion, [Distribution-Based Score Fusion](https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18) scales the scores a bit differently - based on the mean and standard deviation of the scores for each result set.
"""
logger.info("### Distribution-Based Score Fusion")


retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

for node in nodes_with_scores:
    logger.debug(f"Score: {node.score:.2f} - {node.text[:100]}...\n-----")

"""
## Use in a Query Engine!

Now, we can plug our retriever into a query engine to synthesize natural language responses.
"""
logger.info("## Use in a Query Engine!")


query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query("What happened at Interleafe and Viaweb?")


display_response(response)

logger.info("\n\n[DONE]", bright=True)