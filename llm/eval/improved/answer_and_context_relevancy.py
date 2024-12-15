# %pip install nest_asyncio

import os
import joblib
from typing import Sequence, TypedDict
import nest_asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from llama_index.core.llama_dataset import (
    download_llama_dataset,
    BaseLlamaDataset,
    BaseLlamaPredictionDataset,
    BaseLlamaExamplePrediction,
)
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)
from llama_index.core.evaluation.notebook_utils import get_eval_results_df
from IPython.display import display
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage import StorageContext
from llama_index.core.schema import Document, QueryBundle, BaseNode, NodeWithScore
from llama_index.core.indices.base import BaseIndex, BaseQueryEngine
from llama_index.core.node_parser import NodeParser
from llama_index.core import Settings

from jet.llm.ollama import (
    update_llm_settings,
    create_embed_model,
    create_llm,
    small_llm_model,
    large_llm_model,
    large_embed_model,
)
from jet.logger import logger, time_it

nest_asyncio.apply()


class EvalJudges(TypedDict):
    answer_relevancy: AnswerRelevancyEvaluator
    context_relevancy: ContextRelevancyEvaluator


def displayify_df(df):
    """For pretty displaying DataFrame in a notebook."""
    display_df = df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        }
    )
    display(display_df)


@time_it
def load_llm_settings():
    logger.newline()
    logger.debug("Loading LLM settings...")
    settings = update_llm_settings({
        "llm_model": large_llm_model,
        "embedding_model": large_embed_model,
    })
    return settings


@time_it
def download_dataset(cache_dir: str = "./cache", limit: int = None) -> tuple[BaseLlamaDataset, list[Document]]:
    """Download and cache the llama dataset with an optional limit on the number of documents."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "data.pkl")

    # Check if cache exists
    if os.path.exists(cache_path):
        logger.newline()
        logger.debug("Loading dataset from cache...")
        rag_dataset, documents = joblib.load(cache_path)
        return rag_dataset, documents
    else:
        logger.newline()
        logger.debug("Downloading llama dataset...")
        rag_dataset, documents = download_llama_dataset(
            "EvaluatingLlmSurveyPaperDataset", "./data", show_progress=True
        )
        # Save to cache
        logger.debug("Saving dataset to cache...")
        joblib.dump((rag_dataset, documents), cache_path)

    logger.log(
        "rag_dataset.to_pandas()[:5]:",
        rag_dataset.to_pandas()[:5],
        colors=["GRAY", "INFO"]
    )

    # Apply limit if specified
    if limit:
        rag_dataset.examples = rag_dataset.examples[:limit]
        documents = documents[:limit]

    return rag_dataset, documents


@time_it
def create_nodes(parser: NodeParser, documents: Sequence[Document]):
    logger.newline()
    logger.debug("Creating nodes...")
    nodes = parser.get_nodes_from_documents(
        documents, show_progress=True)
    return nodes


@time_it
def create_index(nodes: list[BaseNode]):
    logger.newline()
    logger.debug("Creating index...")
    index = (
        VectorStoreIndex(
            embed_model=create_embed_model(large_embed_model),
            nodes=nodes,
            show_progress=True,
        )
    )
    return index


@time_it
def store_index(index: BaseIndex, storage_dir: str = "./storage/llama_dataset"):
    logger.newline()
    logger.debug("Storing index...")
    index.storage_context.persist(persist_dir=storage_dir)
    return index


@time_it
def load_index(storage_dir: str = "./storage/llama_dataset"):
    logger.newline()
    logger.debug("Loading dataset from cache...")
    storage_context = StorageContext.from_defaults(
        persist_dir=storage_dir
    )
    logger.newline()
    logger.debug("Loading index...")
    index = load_index_from_storage(storage_context)
    return index


@time_it
def create_query_engine(index: BaseIndex, llm_model: str):
    logger.newline()
    logger.debug("Creating query engine...")
    query_engine = index.as_query_engine(llm=create_llm(llm_model))
    return query_engine


@time_it
def make_predictions(dataset: BaseLlamaDataset, query_engine: BaseQueryEngine, batch_size: int = 100) -> BaseLlamaPredictionDataset:
    logger.newline()
    logger.debug("Creating prediction dataset...")
    prediction_dataset = dataset.make_predictions_with(
        predictor=query_engine,
        batch_size=batch_size,
        show_progress=True,
    )
    return prediction_dataset


@time_it
def evaluate_results(judges: EvalJudges, dataset: BaseLlamaDataset, predictions: BaseLlamaExamplePrediction, batch_size: int = 2):
    logger.newline()
    logger.debug("Evaluating correctness...")
    eval_tasks = []
    for example, prediction in zip(dataset.examples, predictions):
        eval_tasks.append(
            judges["answer_relevancy"].evaluate(
                query=example.query,
                response=prediction.response,
                sleep_time_in_seconds=1.0,
            )
        )
        eval_tasks.append(
            judges["context_relevancy"].evaluate(
                query=example.query,
                contexts=prediction.contexts,
                sleep_time_in_seconds=1.0,
            )
        )
    logger.newline()
    logger.debug("Gathering evaluation results...")
    results1 = eval_tasks[:batch_size]
    results2 = eval_tasks[batch_size:]
    return results1 + results2


@time_it
def compute_mean_scores(evals):
    logger.newline()
    logger.debug("Computing mean scores...")
    deep_dfs, mean_dfs = {}, {}
    for metric in evals.keys():
        deep_df, mean_df = get_eval_results_df(
            names=["baseline"] * len(evals[metric]),
            results_arr=evals[metric],
            metric=metric,
        )
        deep_dfs[metric] = deep_df
        mean_dfs[metric] = mean_df
    mean_scores_df = pd.concat(
        [mdf.reset_index() for _, mdf in mean_dfs.items()],
        axis=0,
        ignore_index=True,
    )
    mean_scores_df = mean_scores_df.set_index("index")
    mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])
    return deep_dfs, mean_scores_df


def main():
    settings = load_llm_settings()
    rag_dataset, documents = download_dataset(
        cache_dir="cache/llama_dataset", limit=10)

    if not os.path.exists("./storage/llama_dataset"):
        nodes = create_nodes(settings.node_parser, documents)
        index = create_index(nodes)
        index = store_index(index)
    else:
        index = load_index("./storage/llama_dataset")
    query_engine = create_query_engine(index, large_llm_model)
    prediction_dataset: BaseLlamaPredictionDataset = make_predictions(
        rag_dataset, query_engine)
    judges = {
        "answer_relevancy": AnswerRelevancyEvaluator(llm=create_llm(small_llm_model)),
        "context_relevancy": ContextRelevancyEvaluator(llm=create_llm(large_llm_model)),
    }
    eval_results = evaluate_results(
        judges, rag_dataset, prediction_dataset.predictions)
    evals = {
        "answer_relevancy": eval_results[::2],
        "context_relevancy": eval_results[1::2],
    }
    deep_dfs, mean_scores_df = compute_mean_scores(evals)
    logger.log("mean_scores_df:", mean_scores_df, colors=["LOG", "SUCCESS"])
    displayify_df(deep_dfs["context_relevancy"].head(2))
    cond = deep_dfs["context_relevancy"]["scores"] < 1
    displayify_df(deep_dfs["context_relevancy"][cond].head(5))


if __name__ == "__main__":
    main()
