from IPython.display import Markdown, display
from IPython.display import display, HTML
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import os
import pandas as pd
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
# LLM Reranker Demonstration (Great Gatsby)

This tutorial showcases how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value
in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval
to dynamically select the nodes that are actually relevant to the query.
"""
logger.info("# LLM Reranker Demonstration (Great Gatsby)")

# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Load Data, Build Index
"""
logger.info("## Load Data, Build Index")


Settings.llm = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.chunk_size = 512

documents = SimpleDirectoryReader("../../../examples/gatsby/data").load_data()

documents

index = VectorStoreIndex.from_documents(
    documents,
)

"""
## Retrieval
"""
logger.info("## Retrieval")



pd.set_option("display.max_colwidth", -1)


def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes


def pretty_logger.debug(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_logger.debug(pd.DataFrame(result_dicts))

new_nodes = get_retrieved_nodes(
    "Who was driving the car that hit Myrtle?",
    vector_top_k=3,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "Who was driving the car that hit Myrtle?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What did Gatsby want Daisy to do in front of Tom?",
    vector_top_k=3,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What did Gatsby want Daisy to do in front of Tom?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

"""
## Query Engine
"""
logger.info("## Query Engine")

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        LLMRerank(
            choice_batch_size=5,
            top_n=2,
        )
    ],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do during his time at Y Combinator?",
)

logger.info("\n\n[DONE]", bright=True)