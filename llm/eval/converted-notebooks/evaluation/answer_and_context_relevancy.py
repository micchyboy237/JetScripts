# %pip install nest_asyncio

import os
import joblib
import nest_asyncio
import pandas as pd
import json
from typing import Optional, Sequence, TypedDict
from tqdm import tqdm
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
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core import Settings

from jet.llm.ollama.base import (
    update_llm_settings,
    create_embed_model,
    create_llm,

)
from jet.llm.ollama.constants import (
    OLLAMA_SMALL_LLM_MODEL,
    OLLAMA_LARGE_LLM_MODEL,
    OLLAMA_LARGE_EMBED_MODEL,
)
from jet.transformers import make_serializable
from jet.logger import logger, time_it

nest_asyncio.apply()

ANSWER_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the provided response match the subject matter of the user's query?\n"
    "2. Does the provided response attempt to address the focus or perspective "
    "on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above  "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_int_score>\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)
CONTEXT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources are relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the retrieved context match the subject matter of the user's query?\n"
    "2. Can the retrieved context be used exclusively to provide a full answer to the user's query?\n"
    "Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response "
    "according to the criteria questions previously mentioned. "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the floating number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_score:.2f>\n\n"
    "Query: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Feedback:"
)


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
        "llm_model": OLLAMA_LARGE_LLM_MODEL,
        "embedding_model": OLLAMA_LARGE_EMBED_MODEL,
    })
    return settings


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
            embed_model=create_embed_model(OLLAMA_LARGE_EMBED_MODEL),
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
def download_dataset(
    cache_dir: str = "./cache",
    limit: int = None,
    rag_limit: int = None,
) -> tuple[BaseLlamaDataset, list[Document]]:
    base_name = os.path.splitext(os.path.basename(__file__))[0].lower()
    cache_dir = os.path.join(cache_dir, base_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "data.pkl")

    # Check if cache exists
    if os.path.exists(cache_path):
        logger.newline()
        logger.debug("Loading dataset from cache...")
        rag_dataset, documents = joblib.load(cache_path)
    else:
        logger.newline()
        logger.debug("Downloading llama dataset...")
        rag_dataset, documents = download_llama_dataset(
            "EvaluatingLlmSurveyPaperDataset", "./data", show_progress=True
        )
        # Save to cache
        joblib.dump((rag_dataset, documents), cache_path)
        logger.log("Dataset saved to", cache_path,
                   colors=["WHITE", "BRIGHT_SUCCESS"])

        logger.newline()
        results_dir = os.path.join("./results", base_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "data.json")
        with open(results_path, "w") as f:
            json.dump({
                "documents": make_serializable(documents),
                "rag_examples": make_serializable(rag_dataset.examples),
            }, f, indent=2, ensure_ascii=False)
            logger.log("Saved to", results_path, colors=[
                "WHITE", "BRIGHT_SUCCESS"])

    logger.log(
        "rag_dataset.to_pandas()[:5]:",
        rag_dataset.to_pandas()[:5],
        colors=["GRAY", "INFO"]
    )

    # Apply limit if specified
    if limit:
        documents = documents[:limit]
    if rag_limit:
        rag_dataset.examples = rag_dataset.examples[:rag_limit]

    return rag_dataset, documents


@time_it
def make_predictions(dataset: BaseLlamaDataset, query_engine: BaseQueryEngine, batch_size: int = 1, cache_dir: str = "./cache") -> BaseLlamaPredictionDataset:
    base_name = os.path.splitext(os.path.basename(__file__))[0].lower()
    cache_dir = os.path.join(cache_dir, base_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "predictions.pkl")

    if os.path.exists(cache_path):
        logger.newline()
        prediction_dataset = joblib.load(cache_path)
        logger.log("Loaded prediction results from cache",
                   cache_path, colors=["WHITE", "SUCCESS"])
    else:
        logger.newline()
        logger.debug("Creating prediction dataset...")
        prediction_dataset = dataset.make_predictions_with(
            predictor=query_engine,
            batch_size=batch_size,
            show_progress=True,
        )
        # Save to cache
        joblib.dump(prediction_dataset, cache_path)
        logger.log("Prediction results saved to", cache_path,
                   colors=["WHITE", "BRIGHT_SUCCESS"])

        logger.newline()
        results_dir = os.path.join("./results", base_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "predictions.json")
        with open(results_path, "w") as f:
            json.dump(make_serializable(prediction_dataset.predictions),
                      f, indent=2, ensure_ascii=False)
            logger.log("Saved to", results_path, colors=[
                "WHITE", "BRIGHT_SUCCESS"])

    return prediction_dataset


@time_it
def evaluate_results(judges: EvalJudges, dataset: BaseLlamaDataset, predictions: BaseLlamaExamplePrediction, batch_size: int = 1, cache_dir: str = "./cache", use_cache: bool = False):
    base_name = os.path.splitext(os.path.basename(__file__))[0].lower()
    cache_dir = os.path.join(cache_dir, base_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "eval.pkl")

    if use_cache and os.path.exists(cache_path):
        logger.newline()
        eval_results_dict = joblib.load(cache_path)
        logger.log("Loaded eval results from cache",
                   cache_path, colors=["WHITE", "SUCCESS"])
        eval_tasks = eval_results_dict['eval_results']

    else:
        logger.newline()
        logger.debug("Evaluating correctness...")
        eval_iterator = tqdm(zip(dataset.examples, predictions),
                             total=len(predictions) * batch_size)
        eval_tasks = []
        eval_results_dict = {
            "batch_size": batch_size,
            "dataset": dataset,
            "eval_results": eval_tasks
        }
        for example, prediction in eval_iterator:
            logger.log("Query:", example.query, colors=["GRAY", "DEBUG"])

            logger.debug("Evaluating answer relevancy...")
            answer_relevancy_result = judges["answer_relevancy"].evaluate(
                query=example.query,
                response=prediction.response,
                sleep_time_in_seconds=1.0,
            )

            logger.debug("Evaluating context relevancy...")
            context_relevancy_result = judges["context_relevancy"].evaluate(
                query=example.query,
                contexts=prediction.contexts,
                sleep_time_in_seconds=1.0,
            )

            eval_result = {
                "query": example.query,
                "answer_relevancy": {
                    "response": prediction.response,
                    "result": answer_relevancy_result,
                },
                "context_relevancy": {
                    "contexts": prediction.contexts,
                    "result": context_relevancy_result,
                },
            }
            eval_tasks.append(eval_result)

            # Save to cache
            joblib.dump(eval_results_dict, cache_path)
            logger.log("Eval results saved to", cache_path,
                       colors=["WHITE", "BRIGHT_SUCCESS"])

            logger.newline()
            results_dir = os.path.join("./results", base_name)

            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, "eval.json")
            with open(results_path, "w") as f:
                json.dump(make_serializable(eval_tasks),
                          f, indent=2, ensure_ascii=False)
                logger.log("Saved to", results_path, colors=[
                    "WHITE", "BRIGHT_SUCCESS"])

    return eval_tasks


@time_it
def compute_mean_scores(evals, cache_dir: str = "./cache", use_cache: bool = False):
    base_name = os.path.splitext(os.path.basename(__file__))[0].lower()
    cache_dir = os.path.join(cache_dir, base_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "mean_scores.pkl")

    if use_cache and os.path.exists(cache_path):
        logger.newline()
        deep_dfs, mean_scores_df = joblib.load(cache_path)
        logger.log("Loaded eval results from cache",
                   cache_path, colors=["WHITE", "SUCCESS"])

    else:
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

        # Save to cache
        joblib.dump((deep_dfs, mean_scores_df), cache_path)
        logger.log("Computed mean scores saved to", cache_path,
                   colors=["WHITE", "BRIGHT_SUCCESS"])

        logger.newline()
        results_dir = os.path.join("./results", base_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "mean_scores.json")
        # Convert the DataFrame to a JSON-compatible dictionary
        mean_scores_dict = mean_scores_df.to_dict()
        # Prepare the deep_dfs for JSON serialization if needed
        deep_dfs_serializable = {k: v.to_dict() for k, v in deep_dfs.items()}
        # Combine both into a single structure
        output = {
            "mean_scores": mean_scores_dict,
            "deep_dfs": deep_dfs_serializable,
        }
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            logger.log("Saved to", results_path, colors=[
                "WHITE", "BRIGHT_SUCCESS"])

    return deep_dfs, mean_scores_df


def main():
    settings = load_llm_settings()
    rag_dataset, documents = download_dataset(
        limit=1,
        rag_limit=1,
    )

    if not os.path.exists("./storage/llama_dataset"):
        nodes = create_nodes(settings.node_parser, documents)
        index = create_index(nodes)
        index = store_index(index)
    else:
        index = load_index("./storage/llama_dataset")
    query_engine = create_query_engine(index, OLLAMA_LARGE_LLM_MODEL)

    prediction_dataset: BaseLlamaPredictionDataset = make_predictions(
        rag_dataset, query_engine, batch_size=1)

    judges = {
        "answer_relevancy": AnswerRelevancyEvaluator(
            llm=create_llm(OLLAMA_SMALL_LLM_MODEL),
            eval_template=ANSWER_EVAL_TEMPLATE,
        ),
        "context_relevancy": ContextRelevancyEvaluator(
            llm=create_llm(OLLAMA_LARGE_LLM_MODEL),
            eval_template=CONTEXT_EVAL_TEMPLATE,
        ),
    }
    eval_results = evaluate_results(
        judges, rag_dataset, prediction_dataset.predictions, batch_size=1)
    evals = {
        "answer_relevancy": [eval_result['answer_relevancy']['result'] for eval_result in eval_results],
        "context_relevancy": [eval_result['context_relevancy']['result'] for eval_result in eval_results],
    }
    deep_dfs, mean_scores_df = compute_mean_scores(evals)


if __name__ == "__main__":
    main()
