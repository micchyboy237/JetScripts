from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import LLMRerank
from jet.llm.ollama.base import Ollama
from IPython.display import Markdown, display
import os
from llama_index.core import Settings
from pathlib import Path
import requests
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.postprocessor.rankllm_rerank import RankLLMRerank
import pandas as pd
import torch
from IPython.display import display, HTML

initialize_ollama_settings()

"""
# RankLLM Reranker Demonstration (Van Gogh Wiki)

This demo showcases how to use [RankLLM](https://github.com/castorini/rank_llm) to rerank passages. 

RankLLM offers a suite of listwise rerankers, albeit with focus on open source LLMs finetuned for the task - RankVicuna and RankZephyr being two of them.

It compares query search results from Van Gogh’s wikipedia with just retrieval (using VectorIndexRetriever from llama-index) and retrieval+reranking with RankLLM. We show an example of reranking 50 candidates using the RankZephyr reranker, which uses a listwise sliding window algorithm.


_______________________________
Dependencies:

- **CUDA**
- The built-in retriever, which uses [Pyserini](https://github.com/castorini/pyserini), requires `JDK11`, `PyTorch`, and `Faiss`


### castorini/rank_llm
Suite of LLM-based reranking models (e.g, `RankZephyr`, `LiT5`, `RankVicuna`)
Website: [http://rankllm.ai](http://rankllm.ai)\
"""

# %pip install llama-index-core
# %pip install llama-index-llms-ollama
# %pip install llama-index-postprocessor-rankllm-rerank
# %pip install rank-llm

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# OPENAI_API_KEY = "sk-"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


Settings.llm = Ollama(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.chunk_size = 512

"""
## Load Data, Build Index
"""


wiki_titles = [
    "Vincent van Gogh",
]

data_path = Path("data_wiki")
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

documents = SimpleDirectoryReader("./data_wiki/").load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

"""
### Retrieval + RankLLM Reranking (sliding window)

1. Set up retriever and reranker
2. Retrieve results given search query without reranking
3. Retrieve results given search query with RankZephyr reranking
"""




def get_retrieved_nodes(
    query_str,
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=False,
    model="rank_zephyr",
    window_size=None,
):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)
    retrieved_nodes.reverse()

    if with_reranker:
        reranker = RankLLMRerank(
            model=model, top_n=reranker_top_n, window_size=window_size
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

        del reranker
        torch.cuda.empty_cache()

    return retrieved_nodes


def pretty_logger.debug(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_logger.debug(pd.DataFrame(result_dicts))

"""
### Without `RankZephyr` reranking, the correct result is ranked `47`th/50.

#### Expected result:
```After much pleading from Van Gogh, Gauguin arrived in Arles on 23 October and, in November, the two painted together. Gauguin depicted Van Gogh in his The Painter of Sunflowers;```
"""

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes[:3])

"""
### With `RankZephyr` reranking, the correct result is ranked `1`st/50
"""

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    reranker_top_n=3,
    with_reranker=True,
    model="rank_zephyr",
    window_size=15,
)

visualize_retrieved_nodes(new_nodes)

"""
## Retrieve and Rerank top 10 results using RankVicuna, RankGPT
"""

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
    model="rank_vicuna",
)

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
    model="llama3.2", request_timeout=300.0, context_window=4096,
)

visualize_retrieved_nodes(new_nodes)

"""
#### For other models, use `model=`
- `monot5` for MonoT5 pointwise reranker
- `castorini/LiT5-Distill-base` for LiT5 distill reranker
- `castorini/LiT5-Score-base` for LiT5 score reranker
"""

logger.info("\n\n[DONE]", bright=True)