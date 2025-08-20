import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
load_index_from_storage,
StorageContext,
)
from llama_index.core import Document
from llama_index.core.evaluation import (
SemanticSimilarityEvaluator,
BatchEvalRunner,
)
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.core.evaluation.eval_utils import (
get_responses,
aget_responses,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.param_tuner.base import TunedResult, RunResult
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import IndexNode
from llama_index.experimental.param_tuner import AsyncParamTuner
from llama_index.experimental.param_tuner import ParamTuner
from llama_index.experimental.param_tuner import RayTuneParamTuner
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
import numpy as np
import os
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

"""
# [WIP] Hyperparameter Optimization for RAG

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/param_optimizer/param_optimizer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this guide we show you how to do hyperparameter optimization for RAG.

We use our new, experimental `ParamTuner` class which allows hyperparameter grid search over a RAG function. It comes in two variants:

- `ParamTuner`: a naive way for parameter tuning by iterating over all parameters.
- `RayTuneParamTuner`: a hyperparameter tuning mechanism powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

The `ParamTuner` can take in any function that outputs a dictionary of values. In this setting we define a function that constructs a basic RAG ingestion pipeline from a set of documents (the Llama 2 paper), runs it over an evaluation dataset, and measures a correctness metric.

We investigate tuning the following parameters:

- Chunk size
- Top k value
"""
logger.info("# [WIP] Hyperparameter Optimization for RAG")

# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-ollama
# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-experimental-param-tuner

# !pip install llama-index llama-hub

# !mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"

# import nest_asyncio

# nest_asyncio.apply()

# from llama_index.readers.file import PDFReader
# from llama_index.readers.file import PyMuPDFReader

# loader = PDFReader()
docs0 = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]


"""
## Load "Golden" Evaluation Dataset

Here we setup a "golden" evaluation dataset for the llama2 paper.

**NOTE**: We pull this in from Dropbox. For details on how to generate a dataset please see our `DatasetGenerator` module.
"""
logger.info("## Load "Golden" Evaluation Dataset")

# !wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O data/llama2_eval_qr_dataset.json


eval_dataset = QueryResponseDataset.from_json(
    f"{GENERATED_DIR}/llama2_eval_qr_dataset.json"
)

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

"""
## Define Objective Function + Parameters

Here we define function to optimize given the parameters.

The function specifically does the following: 1) builds an index from documents, 2) queries index, and runs some basic evaluation.
"""
logger.info("## Define Objective Function + Parameters")



"""
### Helper Functions
"""
logger.info("### Helper Functions")

def _build_index(chunk_size, docs):
    index_out_path = f"./storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)

        index = VectorStoreIndex(base_nodes)
        index.storage_context.persist(index_out_path)
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=index_out_path
        )
        index = load_index_from_storage(
            storage_context,
        )
    return index


def _get_eval_batch_runner():
    evaluator_s = SemanticSimilarityEvaluator(embed_model=MLXEmbedding())
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )

    return eval_batch_runner

"""
### Objective Function (Sync)
"""
logger.info("### Objective Function (Sync)")

def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    index = _build_index(chunk_size, docs)

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=True
    )

    eval_batch_runner = _get_eval_batch_runner()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

"""
### Objective Function (Async)
"""
logger.info("### Objective Function (Async)")

async def aobjective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    index = _build_index(chunk_size, docs)

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    async def async_func_11():
        pred_response_objs = get_responses(
            eval_qs, query_engine, show_progress=True
        )
        return pred_response_objs
    pred_response_objs = asyncio.run(async_func_11())
    logger.success(format_json(pred_response_objs))

    eval_batch_runner = _get_eval_batch_runner()
    async def async_func_16():
        eval_results = eval_batch_runner.evaluate_responses(
            eval_qs, responses=pred_response_objs, reference=ref_response_strs
        )
        return eval_results
    eval_results = asyncio.run(async_func_16())
    logger.success(format_json(eval_results))

    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

"""
### Parameters

We define both the parameters to grid-search over `param_dict` and fixed parameters `fixed_param_dict`.
"""
logger.info("### Parameters")

param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}
fixed_param_dict = {
    "docs": docs,
    "eval_qs": eval_qs[:10],
    "ref_response_strs": ref_response_strs[:10],
}

"""
## Run ParamTuner (default)

Here we run our default param tuner, which iterates through all hyperparameter combinations either synchronously or in async.
"""
logger.info("## Run ParamTuner (default)")


param_tuner = ParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)

results = param_tuner.tune()

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
logger.debug(f"Score: {best_result.score}")
logger.debug(f"Top-k: {best_top_k}")
logger.debug(f"Chunk size: {best_chunk_size}")

test_idx = 6
p = results.run_results[test_idx].params
(results.run_results[test_idx].score, p["top_k"], p["chunk_size"])

"""
### Run ParamTuner (Async)

Run the async version.
"""
logger.info("### Run ParamTuner (Async)")


aparam_tuner = AsyncParamTuner(
    aparam_fn=aobjective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    num_workers=2,
    show_progress=True,
)

async def run_async_code_4f7573a1():
    async def run_async_code_cfd003fd():
        results = await aparam_tuner.atune()
        return results
    results = asyncio.run(run_async_code_cfd003fd())
    logger.success(format_json(results))
    return results
results = asyncio.run(run_async_code_4f7573a1())
logger.success(format_json(results))

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
logger.debug(f"Score: {best_result.score}")
logger.debug(f"Top-k: {best_top_k}")
logger.debug(f"Chunk size: {best_chunk_size}")

"""
## Run ParamTuner (Ray Tune)

Here we run our tuner powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), a library for scalable hyperparameter tuning.

In the notebook we run it locally, but you can run this on a cluster as well.
"""
logger.info("## Run ParamTuner (Ray Tune)")


param_tuner = RayTuneParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    run_config_dict={"storage_path": "/tmp/custom/ray_tune", "name": "my_exp"},
)

results = param_tuner.tune()

results.best_run_result.params.keys()

results.best_idx

best_result = results.best_run_result

best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
logger.debug(f"Score: {best_result.score}")
logger.debug(f"Top-k: {best_top_k}")
logger.debug(f"Chunk size: {best_chunk_size}")

logger.info("\n\n[DONE]", bright=True)