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
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
import pandas as pd
from IPython.display import display, HTML
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle

initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/rankGPT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# RankGPT Reranker Demonstration (Van Gogh Wiki)

This demo integrates [RankGPT](https://github.com/sunnweiwei/RankGPT) into LlamaIndex as a reranker.

Paper: [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542)

the idea of `RankGPT`:
* it is a zero-shot listwise passage reranking using LLM (ChatGPT or GPT-4 or other LLMs)
* it applies permutation generation approach and sliding window strategy to rerank passages efficiently. 

In this example, we use Van Gogh's wikipedia as an example to compare the Retrieval results with/without RankGPT reranking.
we showcase two models for RankGPT:
* Ollama `GPT3.5`
* `Mistral` model.
"""

# %pip install llama-index-postprocessor-rankgpt-rerank
# %pip install llama-index-llms-huggingface
# %pip install llama-index-llms-huggingface-api
# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# OPENAI_API_KEY = "sk-"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## Load Data, Build Index
"""


Settings.llm = Ollama(temperature=0, model="llama3.2",
                      request_timeout=300.0, context_window=4096)
Settings.chunk_size = 512

"""
### Download Van Gogh wiki from Wikipedia
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

"""
### Build vector store index for this Wikipedia page
"""

index = VectorStoreIndex.from_documents(
    documents,
)

"""
## Retrieval + RankGPT reranking
Steps:
1. Setting up retriever and reranker (as an option) 
2. Retrieve results given a search query without reranking
3. Retrieve results given a search query with RankGPT reranking enabled
4. Comparing the results with and without reranking
"""


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
        reranker = RankGPTRerank(
            llm=Ollama(
                model="llama3.2", request_timeout=300.0, context_window=4096,
                temperature=0.0,
                #                 api_key=OPENAI_API_KEY,
            ),
            top_n=reranker_top_n,
            verbose=True,
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


"""
### Retrieval top 3 results without Reranking
"""

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=3,
    with_reranker=False,
)

"""
### Expected result is:
```After much pleading from Van Gogh, Gauguin arrived in Arles on 23 October and, in November, the two painted together. Gauguin depicted Van Gogh in his The Painter of Sunflowers;```
"""

visualize_retrieved_nodes(new_nodes)

"""
#### Finding: the right result is ranked at 2nd without reranking

### Retrieve and Reranking top 10 results using RankGPT and return top 3
"""

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles ?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

"""
#### Finding: After RankGPT reranking, the top 1st result is the right text containing the answer

## Using other LLM for RankGPT reranking

### Using `Ollama` for serving local `Mistral` models
"""


llm = Ollama(model="mistral", request_timeout=30.0)


def get_retrieved_nodes(
    query_str, vector_top_k=5, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        reranker = RankGPTRerank(
            llm=llm,
            top_n=reranker_top_n,
            verbose=True,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes


new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles ?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

logger.info("\n\n[DONE]", bright=True)
