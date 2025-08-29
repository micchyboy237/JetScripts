from IPython.display import Markdown, display
from IPython.display import display, HTML
from copy import deepcopy
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
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

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/LLMReranker-Lyft-10k.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LLM Reranker Demonstration (2021 Lyft 10-k)

This tutorial showcases how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value
in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval
to dynamically select the nodes that are actually relevant to the query.
"""
logger.info("# LLM Reranker Demonstration (2021 Lyft 10-k)")

# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
## Load Data, Build Index
"""
logger.info("## Load Data, Build Index")


Settings.llm = OllamaFunctionCallingAdapter(
    temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

Settings.chunk_overlap = 0
Settings.chunk_size = 128

documents = SimpleDirectoryReader(
    input_files=[f"{os.path.dirname(__file__)}/data/10k/lyft_2021.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

"""
## Retrieval Comparisons
"""
logger.info("## Retrieval Comparisons")


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
        node = deepcopy(node)
        node.node.metadata = None
        node_text = node.node.get_text()
        node_text = node_text.replace("\n", " ")

        result_dict = {"Score": node.score, "Text": node_text}
        result_dicts.append(result_dict)

    pretty_logger.debug(pd.DataFrame(result_dicts))


new_nodes = get_retrieved_nodes(
    "What is Lyft's response to COVID-19?", vector_top_k=5, with_reranker=False
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What is Lyft's response to COVID-19?",
    vector_top_k=20,
    reranker_top_n=5,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What initiatives are the company focusing on independently of COVID-19?",
    vector_top_k=5,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What initiatives are the company focusing on independently of COVID-19?",
    vector_top_k=40,
    reranker_top_n=5,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

logger.info("\n\n[DONE]", bright=True)
