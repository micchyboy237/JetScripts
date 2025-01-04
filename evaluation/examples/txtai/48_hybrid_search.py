from dataclasses import dataclass
import os
import time
import traceback
from typing import Optional, Union, TypedDict, Any
from tqdm import tqdm
from jet.llm.benchmarks import evaluate
from jet.logger import logger

BASE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/evaluation"
BEIR_DIR = os.path.join(BASE_DIR, "beir")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
datasets = ["nfcorpus", "fiqa", "arguana", "scidocs", "scifact"]
methods = ["bm25", "embed", "hybrid"]


class Config(TypedDict, total=False):
    """
    Configuration for the embeddings index.
    """
    # General configuration
    path: Optional[str]  # Path to vectors model or index
    # Number of PCA components for dimensionality reduction
    pca: Optional[int]
    dimensions: Optional[int]  # Number of embedding dimensions
    content: Optional[bool]  # Whether content storage is enabled
    objects: Optional[dict[str, Any]]  # Object-specific configuration
    storevectors: Optional[bool]  # Whether to store vector models
    # Scoring configuration (method, terms, normalization)
    scoring: Optional[Union[str, dict[str, Any]]]
    query: Optional[dict[str, Any]]  # Query-specific configuration
    graph: Optional[dict[str, Any]]  # Graph-specific configuration
    indexes: Optional[dict[str, "Config"]]  # Sub-index configurations
    # Function-specific configuration
    functions: Optional[dict[str, Any]]
    # Custom column definitions for text/object
    columns: Optional[dict[str, Any]]
    # Shortcut for enabling keyword-based scoring
    keyword: Optional[bool]
    hybrid: Optional[bool]  # Shortcut for enabling hybrid scoring
    defaults: Optional[bool]  # Whether to allow default configurations


@dataclass
class EvaluationArgs:
    directory: str
    name: str
    methods: str
    topk: int
    output: str
    config: Config  # Use default_factory with default config
    refresh: bool = False


def run_evaluations():
    """
    Executes evaluations for each method and dataset using the `evaluate` function.
    Tracks progress, logs results, and handles errors.
    """
    total_evaluations = len(methods) * len(datasets)
    progress_bar = tqdm(total=total_evaluations, desc="Evaluations Progress")

    for method in methods:
        for dataset in datasets:
            start_time = time.time()
            dataset_dir = os.path.join(BEIR_DIR, dataset)
            config_dir = os.path.join(dataset_dir, method)
            config_file = os.path.join(config_dir, "config")
            os.makedirs(config_dir, exist_ok=True)

            args = EvaluationArgs(
                directory=BEIR_DIR,
                name=dataset,
                methods=method,
                topk=10,
                output=OUTPUT_DIR,
                config=config_file if os.path.exists(
                    config_file) else Config(),
            )

            try:
                logger.log("Evaluating: Method:", method, "Dataset:", dataset,
                           colors=["GRAY", "DEBUG", "GRAY", "DEBUG"])

                performance = evaluate(
                    [method], dataset_dir, args)
                elapsed_time = time.time() - start_time

                logger.log("Performance:", performance,
                           colors=["GRAY", "SUCCESS"])
                logger.log("Time:", f"{elapsed_time:.2f}s",
                           colors=["GRAY", "SUCCESS"])
            except Exception as e:
                logger.error(f"Failed: Method={method}, Dataset={
                             dataset}, Error={str(e)}")
                traceback.print_exc()  # Print the full traceback to the console
            progress_bar.update(1)

    progress_bar.close()
    print("All evaluations completed!")


if __name__ == "__main__":
    run_evaluations()
