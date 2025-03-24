from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/relative_score_dist_fusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Relative Score Fusion and Distribution-Based Score Fusion
#
# In this example, we demonstrate using QueryFusionRetriever with two methods which aim to improve on Reciprocal Rank Fusion:
# 1. Relative Score Fusion ([Weaviate](https://weaviate.io/blog/hybrid-search-fusion-algorithms))
# 2. Distribution-Based Score Fusion ([Mazzeschi: blog post](https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18))

# %pip install llama-index-llms-ollama
# %pip install llama-index-retrievers-bm25

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

# Setup

#
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/summaries/jet-resume", required_exts=[".md"]).load_data()
query = "Tell me about yourself."
# Next, we will setup a vector index over the documentation.


splitter = SentenceSplitter(chunk_size=256)
total_nodes = len(splitter.get_nodes_from_documents(documents))
initial_similarity_k = len(documents)
final_similarity_k = total_nodes

index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], show_progress=True
)

# Create a Hybrid Fusion Retriever using Relative Score Fusion
#
# In this step, we fuse our index with a BM25 based retriever. This will enable us to capture both semantic relations and keywords in our input queries.
#
# Since both of these retrievers calculate a score, we can use the `QueryFusionRetriever` to re-sort our nodes without using an additional models or excessive computation.
#
# The following example uses the [Relative Score Fusion](https://weaviate.io/blog/hybrid-search-fusion-algorithms) algorithm from Weaviate, which applies a MinMax scaler to each result set, then makes a weighted sum. Here, we'll give the vector retriever slightly more weight than BM25 (0.6 vs. 0.4).


def setup_retrievers():
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = index.as_retriever(
        similarity_top_k=initial_similarity_k)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=final_similarity_k
    )

    return [vector_retriever, bm25_retriever]


def main_relative_score_fusion():
    # First, we create our retrievers. Each will retrieve the top-10 most similar nodes.
    retrievers = setup_retrievers()

    # Next, we can create our fusion retriever, which well return the top-10 most similar nodes from the 20 returned nodes from the retrievers.
    #
    # Note that the vector and BM25 retrievers may have returned all the same nodes, only in different orders; in this case, it simply acts as a re-ranker.

    from llama_index.core.retrievers import QueryFusionRetriever

    retriever = QueryFusionRetriever(
        retrievers,
        retriever_weights=[0.6, 0.4],
        similarity_top_k=final_similarity_k,
        num_queries=1,  # set this to 1 to disable query generation
        mode="relative_score",
        use_async=True,
        verbose=True,
    )

    nodes_with_scores = retriever.retrieve(query)
    display_jet_source_nodes(query, nodes_with_scores)
    return retriever

# Distribution-Based Score Fusion
#
# A variant on Relative Score Fusion, [Distribution-Based Score Fusion](https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18) scales the scores a bit differently - based on the mean and standard deviation of the scores for each result set.


def main_distribution_based_score_fusion():
    # First, we create our retrievers. Each will retrieve the top-10 most similar nodes.
    retrievers = setup_retrievers()

    from llama_index.core.retrievers import QueryFusionRetriever

    retriever = QueryFusionRetriever(
        retrievers,
        retriever_weights=[0.6, 0.4],
        similarity_top_k=final_similarity_k,
        num_queries=1,  # set this to 1 to disable query generation
        mode="dist_based_score",
        use_async=True,
        verbose=True,
    )

    nodes_with_scores = retriever.retrieve(query)
    display_jet_source_nodes(query, nodes_with_scores)

    return retriever


def query_nodes(query, retriever):
    # Use in a Query Engine!
    #
    # Now, we can plug our retriever into a query engine to synthesize natural language responses.

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)

    display_jet_source_nodes(query, response)

    return response


if __name__ == "__main__":
    logger.info("main_relative_score_fusion: retriever...")
    retriever = main_relative_score_fusion()
    logger.newline()
    logger.info("main_relative_score_fusion: query...")
    response = query_nodes(query, retriever)

    logger.newline()
    logger.newline()

    logger.info("main_distribution_based_score_fusion: retriever...")
    retriever = main_distribution_based_score_fusion()
    logger.newline()
    logger.info("main_distribution_based_score_fusion: query...")
    response = query_nodes(query, retriever)

    logger.info("\n\n[DONE]", bright=True)
