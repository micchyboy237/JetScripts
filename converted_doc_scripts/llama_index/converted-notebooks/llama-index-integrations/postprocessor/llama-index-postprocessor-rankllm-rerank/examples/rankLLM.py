from IPython.display import Markdown, display
from IPython.display import display, HTML
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.rankLLM_rerank import RankLLMRerank
from pathlib import Path
import logging
import os
import pandas as pd
import requests
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# RankLLM Reranker Demonstration (Van Gogh Wiki)

This demo showcases how to use RankLLM (https://github.com/castorini/rank_llm) to rerank passages. RankLLM offers a suite of listwise, pairwise, and pointwise rerankers, albeit with focus on open source LLMs finetuned for the task - RankVicuna and RankZephyr being two of them. It also features ranking with OllamaFunctionCallingAdapter and GenAI.

It compares query search results from Van Gogh’s wikipedia with just retrieval (using VectorIndexRetriever from llama-index) and retrieval+reranking with RankLLM. It demonstrates two models from RankLLM:

- ```RankVicuna 7B V1```
- ```RankZephyr 7B V1 - Full - BF16```

Dependencies:

- RankLLM's [dependencies](https://github.com/castorini/rank_llm?tab=readme-ov-file#-installation)
- The built-in retriever, which uses [Pyserini](https://github.com/castorini/pyserini), requires `JDK11`, `PyTorch`, and `Faiss`


### castorini/rank_llm
RankLLM is a Python toolkit for reproducible information retrieval research using rerankers, with a focus on listwise reranking.\
Website: [http://rankllm.ai](http://rankllm.ai)\
Stars: 448
"""
logger.info("# RankLLM Reranker Demonstration (Van Gogh Wiki)")

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


Settings.llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.chunk_size = 512

"""
## Load Data, Build Index
"""
logger.info("## Load Data, Build Index")


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
## Retrieval + RankLLM Reranking

1. Set up retriever and reranker
2. Retrieve results given search query without reranking
3. Retrieve results given search query with RankZephyr reranking
4. Retrieve results given search query with RankVicuna reranking

**All the field arguments and defaults for RankLLMRerank:**
```python
model: str = Field(
  description="Model name.",
  default="rank_zephyr"
)
top_n: Optional[int] = Field(
  description="Number of nodes to return sorted by reranking score."
)
window_size: int = Field(
  description="Reranking window size. Applicable only for listwise and pairwise models.",
  default=20
)
batch_size: Optional[int] = Field(
  description="Reranking batch size. Applicable only for pointwise models."
)
context_size: int = Field(
  description="Maximum number of tokens for the context window.",
  default=4096
)
prompt_mode: PromptMode = Field(
  description="Prompt format and strategy used when invoking the reranking model.",
  default=PromptMode.RANK_GPT
)
num_gpus: int = Field(
  description="Number of GPUs to use for inference if applicable.",
  default=1
)
num_few_shot_examples: int = Field(
  description="Number of few-shot examples to include in the prompt.",
  default=0
)
few_shot_file: Optional[str] = Field(
  description="Path to a file containing few-shot examples, used if few-shot prompting is enabled.",
  default=None
)
use_logits: bool = Field(
  description="Whether to use raw logits for reranking scores instead of probabilities.",
  default=False
)
use_alpha: bool = Field(
  description="Whether to apply an alpha scaling factor in the reranking score calculation.",
  default=False
)
variable_passages: bool = Field(
  description="Whether to allow passages of variable lengths instead of fixed-size chunks.",
  default=False
)
stride: int = Field(
  description="Stride to use when sliding over long documents for reranking.",
  default=10
)
use_azure_ollama: bool = Field(
  description="Whether to use Azure OllamaFunctionCallingAdapter instead of the standard OllamaFunctionCallingAdapter API.",
  default=False
)
```
"""
logger.info("## Retrieval + RankLLM Reranking")




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
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

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
## Retrieval top 3 results without reranking

## Expected result:
```After much pleading from Van Gogh, Gauguin arrived in Arles on 23 October and, in November, the two painted together. Gauguin depicted Van Gogh in his The Painter of Sunflowers;```
"""
logger.info("## Retrieval top 3 results without reranking")

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

"""
### The correct result is ranked 3rd.

## Retrieve and Rerank top 10 results using RankZephyr and return top 3
"""
logger.info("### The correct result is ranked 3rd.")

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
### The correct result is ranked 1st after RankZephyr rerank.

## Retrieve and Rerank top 10 results using RankVicuna and return top 3.
"""
logger.info("### The correct result is ranked 1st after RankZephyr rerank.")

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
    model="rank_vicuna",
)

visualize_retrieved_nodes(new_nodes)

"""
### The correct result is ranked 1st after RankVicuna rerank.

## Retrieve and Rerank top 10 results using RankGPT and return top 3

RankGPT is built into RankLLM and can be used as shown below.
"""
logger.info("### The correct result is ranked 1st after RankVicuna rerank.")

new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
    model="llama3.2",
)

visualize_retrieved_nodes(new_nodes)

"""
### The correct result is ranked 1st after RankGPT rerank.

## Sliding window example with RankZephyr.
"""
logger.info("### The correct result is ranked 1st after RankGPT rerank.")




def get_retrieved_nodes_mixed(
    query_str,
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=False,
    step_size=10,
    model="rank_zephyr",
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
            top_n=reranker_top_n,
            model=model,
        )
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    return retrieved_nodes

"""
### After retrieving the top 50 results and reversing the order, the correct result is ranked 47th/50.

### Expected result:
```After much pleading from Van Gogh, Gauguin arrived in Arles on 23 October and, in November, the two painted together. Gauguin depicted Van Gogh in his The Painter of Sunflowers;```
"""
logger.info("### After retrieving the top 50 results and reversing the order, the correct result is ranked 47th/50.")

new_nodes = get_retrieved_nodes_mixed(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

"""
### Retrieve and Rerank reversed top 50 results using RankZephyr and return top 3

The sliding window size is 20, with a step size of 10.
"""
logger.info("### Retrieve and Rerank reversed top 50 results using RankZephyr and return top 3")

new_nodes = get_retrieved_nodes_mixed(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    reranker_top_n=3,
    with_reranker=True,
    model="rank_zephyr",
    step_size=10,
)

visualize_retrieved_nodes(new_nodes)

"""
### The correct result is ranked 1st/50 after RankZephyr rerank.
"""
logger.info("### The correct result is ranked 1st/50 after RankZephyr rerank.")

logger.info("\n\n[DONE]", bright=True)