import asyncio
from jet.transformers.formatters import format_json
from collections import defaultdict
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import (
CorrectnessEvaluator,
SemanticSimilarityEvaluator,
RelevancyEvaluator,
FaithfulnessEvaluator,
PairwiseComparisonEvaluator,
)
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.evaluation.eval_utils import (
get_responses,
get_results_df,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import numpy as np
import openai
import os
import pandas as pd
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/MetadataReplacementDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Metadata Replacement + Node Sentence Window

In this notebook, we use the `SentenceWindowNodeParser` to parse documents into single sentences per node. Each node also contains a "window" with the sentences on either side of the node sentence.

Then, after retrieval, before passing the retrieved sentences to the LLM, the single sentences are replaced with a window containing the surrounding sentences using the `MetadataReplacementNodePostProcessor`.

This is most useful for large documents/indexes, as it helps to retrieve more fine-grained details.

By default, the sentence window is 5 sentences on either side of the original sentence.

In this case, chunk size settings are not used, in favor of following the window settings.
"""
logger.info("# Metadata Replacement + Node Sentence Window")

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

# %load_ext autoreload
# %autoreload 2

"""
## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("## Setup")

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

text_splitter = SentenceSplitter()

llm = MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)


Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

"""
## Load Data, Build the Index

In this section, we load data and build the vector index.

### Load Data

Here, we build an index using chapter 3 of the recent IPCC climate report.
"""
logger.info("## Load Data, Build the Index")

# !curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf


documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

"""
### Extract Nodes

We extract out the set of nodes that will be stored in the VectorIndex. This includes both the nodes with the sentence window parser, as well as the "base" nodes extracted using the standard parser.
"""
logger.info("### Extract Nodes")

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes = text_splitter.get_nodes_from_documents(documents)

"""
### Build the Indexes

We build both the sentence index, as well as the "base" index (with default chunk sizes).
"""
logger.info("### Build the Indexes")


sentence_index = VectorStoreIndex(nodes)

base_index = VectorStoreIndex(base_nodes)

"""
## Querying

### With MetadataReplacementPostProcessor

Here, we now use the `MetadataReplacementPostProcessor` to replace the sentence in each node with it's surrounding context.
"""
logger.info("## Querying")


query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
logger.debug(window_response)

"""
We can also check the original sentence that was retrieved for each node, as well as the actual window of sentences that was sent to the LLM.
"""
logger.info("We can also check the original sentence that was retrieved for each node, as well as the actual window of sentences that was sent to the LLM.")

window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

logger.debug(f"Window: {window}")
logger.debug("------------------")
logger.debug(f"Original Sentence: {sentence}")

"""
### Contrast with normal VectorStoreIndex
"""
logger.info("### Contrast with normal VectorStoreIndex")

query_engine = base_index.as_query_engine(similarity_top_k=2)
vector_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
logger.debug(vector_response)

"""
Well, that didn't work. Let's bump up the top k! This will be slower and use more tokens compared to the sentence window index.
"""
logger.info("Well, that didn't work. Let's bump up the top k! This will be slower and use more tokens compared to the sentence window index.")

query_engine = base_index.as_query_engine(similarity_top_k=5)
vector_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
logger.debug(vector_response)

"""
## Analysis

So the `SentenceWindowNodeParser` + `MetadataReplacementNodePostProcessor` combo is the clear winner here. But why?

Embeddings at a sentence level seem to capture more fine-grained details, like the word `AMOC`.

We can also compare the retrieved chunks for each index!
"""
logger.info("## Analysis")

for source_node in window_response.source_nodes:
    logger.debug(source_node.node.metadata["original_text"])
    logger.debug("--------")

"""
Here, we can see that the sentence window index easily retrieved two nodes that talk about AMOC. Remember, the embeddings are based purely on the original sentence here, but the LLM actually ends up reading the surrounding context as well!

Now, let's try and disect why the naive vector index failed.
"""
logger.info("Here, we can see that the sentence window index easily retrieved two nodes that talk about AMOC. Remember, the embeddings are based purely on the original sentence here, but the LLM actually ends up reading the surrounding context as well!")

for node in vector_response.source_nodes:
    logger.debug("AMOC mentioned?", "AMOC" in node.node.text)
    logger.debug("--------")

"""
So source node at index [2] mentions AMOC, but what did this text actually look like?
"""
logger.info("So source node at index [2] mentions AMOC, but what did this text actually look like?")

logger.debug(vector_response.source_nodes[2].node.text)

"""
So AMOC is disuccsed, but sadly it is in the middle chunk. With LLMs, it is often observed that text in the middle of retrieved context is often ignored or less useful. A recent paper ["Lost in the Middle" discusses this here](https://arxiv.org/abs/2307.03172).

## [Optional] Evaluation

We more rigorously evaluate how well the sentence window retriever works compared to the base retriever.

We define/load an eval benchmark dataset and then run different evaluations over it.

**WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.
"""
logger.info("## [Optional] Evaluation")


# import nest_asyncio

# nest_asyncio.apply()

len(base_nodes)

num_nodes_eval = 30
sample_eval_nodes = random.sample(base_nodes[:200], num_nodes_eval)
dataset_generator = DatasetGenerator(
    sample_eval_nodes,
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
    show_progress=True,
    num_questions_per_chunk=2,
)

async def run_async_code_33724ba0():
    async def run_async_code_12567cda():
        eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()
        return eval_dataset
    eval_dataset = asyncio.run(run_async_code_12567cda())
    logger.success(format_json(eval_dataset))
    return eval_dataset
eval_dataset = asyncio.run(run_async_code_33724ba0())
logger.success(format_json(eval_dataset))

eval_dataset.save_json(f"{GENERATED_DIR}/ipcc_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json(f"{GENERATED_DIR}/ipcc_eval_qr_dataset.json")

"""
### Compare Results
"""
logger.info("### Compare Results")

# import nest_asyncio

# nest_asyncio.apply()




evaluator_c = CorrectnessEvaluator(llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"))
evaluator_s = SemanticSimilarityEvaluator()
evaluator_r = RelevancyEvaluator(llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"))
evaluator_f = FaithfulnessEvaluator(llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"))


max_samples = 30

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

base_query_engine = base_index.as_query_engine(similarity_top_k=2)
query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)


base_pred_responses = get_responses(
    eval_qs[:max_samples], base_query_engine, show_progress=True
)
pred_responses = get_responses(
    eval_qs[:max_samples], query_engine, show_progress=True
)

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

"""
Run evaluations over faithfulness/semantic similarity.
"""
logger.info("Run evaluations over faithfulness/semantic similarity.")

async def async_func_0():
    eval_results = batch_runner.evaluate_responses(
        queries=eval_qs[:max_samples],
        responses=pred_responses[:max_samples],
        reference=ref_response_strs[:max_samples],
    )
    return eval_results
eval_results = asyncio.run(async_func_0())
logger.success(format_json(eval_results))

async def async_func_6():
    base_eval_results = batch_runner.evaluate_responses(
        queries=eval_qs[:max_samples],
        responses=base_pred_responses[:max_samples],
        reference=ref_response_strs[:max_samples],
    )
    return base_eval_results
base_eval_results = asyncio.run(async_func_6())
logger.success(format_json(base_eval_results))

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Sentence Window Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
display(results_df)

logger.info("\n\n[DONE]", bright=True)