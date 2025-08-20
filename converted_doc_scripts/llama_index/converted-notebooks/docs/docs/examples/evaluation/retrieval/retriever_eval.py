import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
generate_question_context_pairs,
EmbeddingQAFinetuneDataset,
)
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import pandas as pd
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/retrieval/retriever_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Retrieval Evaluation

This notebook uses our `RetrieverEvaluator` to evaluate the quality of any Retriever module defined in LlamaIndex.

We specify a set of different evaluation metrics: this includes hit-rate, MRR, Precision, Recall, AP, and NDCG. For any given question, these will compare the quality of retrieved results from the ground-truth context.

To ease the burden of creating the eval dataset in the first place, we can rely on synthetic data generation.

## Setup

Here we load in data (PG essay), parse into Nodes. We then index this data using our simple vector index and get a retriever.
"""
logger.info("# Retrieval Evaluation")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file

# import nest_asyncio

# nest_asyncio.apply()


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

vector_index = VectorStoreIndex(nodes)
retriever = vector_index.as_retriever(similarity_top_k=2)

"""
### Try out Retrieval

We'll try out retrieval over a simple dataset.
"""
logger.info("### Try out Retrieval")

retrieved_nodes = retriever.retrieve("What did the author do growing up?")


for node in retrieved_nodes:
    display_source_node(node, source_length=1000)

"""
## Build an Evaluation dataset of (query, context) pairs

Here we build a simple evaluation dataset over the existing text corpus.

We use our `generate_question_context_pairs` to generate a set of (question, context) pairs over a given unstructured text corpus. This uses the LLM to auto-generate questions from each context chunk.

We get back a `EmbeddingQAFinetuneDataset` object. At a high-level this contains a set of ids mapping to queries and relevant doc chunks, as well as the corpus itself.
"""
logger.info("## Build an Evaluation dataset of (query, context) pairs")


qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
logger.debug(list(queries)[2])

qa_dataset.save_json("pg_eval_dataset.json")

qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset.json")

"""
## Use `RetrieverEvaluator` for Retrieval Evaluation

We're now ready to run our retrieval evals. We'll run our `RetrieverEvaluator` over the eval dataset that we generated.

We define two functions: `get_eval_results` and also `display_results` that run our retriever over the dataset.
"""
logger.info("## Use `RetrieverEvaluator` for Retrieval Evaluation")

include_cohere_rerank = False

if include_cohere_rerank:
#     !pip install cohere -q


metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

if include_cohere_rerank:
    metrics.append(
        "cohere_rerank_relevancy"  # requires COHERE_API_KEY environment variable to be set
    )

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)

sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
logger.debug(eval_result)

async def run_async_code_6d50bb40():
    async def run_async_code_86bbaaaf():
        eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)
        return eval_results
    eval_results = asyncio.run(run_async_code_86bbaaaf())
    logger.success(format_json(eval_results))
    return eval_results
eval_results = asyncio.run(run_async_code_6d50bb40())
logger.success(format_json(eval_results))



def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }

    if include_cohere_rerank:
        crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
        columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

    metric_df = pd.DataFrame(columns)

    return metric_df

display_results("top-2 eval", eval_results)

logger.info("\n\n[DONE]", bright=True)