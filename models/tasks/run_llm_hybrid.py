import os
import time
from typing import List, Union
from llama_cpp import Llama

from jet.data.utils import generate_key
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.tasks.llm_search import search_docs
from jet.models.tasks.llm_rerank import rerank_docs
from jet.models.tokenizer.utils import calculate_n_ctx
from jet.models.utils import get_embedding_size
from jet.models.tasks.task_types import SimilarityResult
from jet.vectors.document_types import HeaderDocument


def map_search_results_to_docs(search_results: List[SimilarityResult], docs: List[HeaderDocument]) -> List[HeaderDocument]:
    """Map search results back to original documents.

    Args:
        search_results: List of search results with document IDs
        docs: Original list of documents to map back to

    Returns:
        List of documents that matched the search results
    """
    result_ids = {result["id"] for result in search_results}
    return [doc for doc in docs if doc["id"] in result_ids]


if __name__ == "__main__":
    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    query = "List all ongoing and upcoming isekai anime 2025."
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        query
    ]
    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            doc["metadata"]["content"],
        ]).strip()
        for doc in docs
    ]
    ids = [doc.id for doc in docs]

    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/8aa0010e73a1075e99dfc213a475a60fd971bbe7/Qwen3-Embedding-0.6B-f16.gguf"
    model_name = "mlx-community/Qwen3-0.6B-4bit"

    n_ctx = calculate_n_ctx(model_name, documents)

    settings = {
        "model_path": model_path,
        "embedding": True,
        "n_ctx": n_ctx,
        "n_threads": 8,
        "n_gpu_layers": -1,
        "n_threads_batch": 64,
        "no_perf": True,      # Disable performance timings
        "verbose": True,
        "flash_attn": True,
        # "n_batch": 512,
        # "n_ubatch": 512,

    }
    key = generate_key(**settings)
    model = Llama(**settings)

    output_key_dir = os.path.join(output_dir, key)

    try:
        # Initial search
        print("Starting search docs...")
        search_start_time = time.time()
        search_results = search_docs(
            model, queries, documents, task, ids=ids, threshold=0.5)
        search_results = search_results[0]
        for res in search_results:
            print(f"\nQuery: {query} (Search Results)")
            print(
                f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
        search_execution_time = time.time() - search_start_time
        save_file({
            "execution_time": f"{search_execution_time:.2f}s",
            "query": query,
            "count": len(search_results),
            "results": search_results
        }, f"{output_key_dir}/search_results.json")
        mapped_search_results_to_docs = map_search_results_to_docs(
            search_results, docs)

        search_result_documents = [
            "\n".join([
                doc["metadata"].get("parent_header") or "",
                doc["metadata"]["header"],
                doc["metadata"]["content"],
            ]).strip()
            for doc in mapped_search_results_to_docs
        ]
        ids = [doc.id for doc in mapped_search_results_to_docs]

        # Rerank results
        print("Starting rerank docs...")
        rerank_start_time = time.time()
        rerank_results = rerank_docs(
            query, search_result_documents, task, ids=ids, show_progress=True)
        rerank_results = rerank_results[0]
        for res in rerank_results:
            print(f"\nQuery: {query} (Reranked Results)")
            print(
                f"Rank: {res['ran']}, Score: {res['score']:.4f}, Text: {res['text']}")
        rerank_execution_time = time.time() - rerank_start_time

        save_file({
            "search_execution_time": f"{search_execution_time:.2f}s",
            "rerank_execution_time": f"{rerank_execution_time:.2f}s",
            "total_execution_time": f"{(search_execution_time + rerank_execution_time):.2f}s",
            "query": query,
            "count": len(rerank_results),
            "results": rerank_results
        }, f"{output_key_dir}/rerank_results.json")
        save_file(settings, f"{output_key_dir}/settings.json")

        print("n_ctx:", n_ctx)

        print("\nExecution Times:")
        logger.log(f"Search Execution Time:",
                   f"{search_execution_time:.2f}s", colors=["GRAY", "ORANGE"])
        logger.log(f"Rerank Execution Time:",
                   f"{rerank_execution_time:.2f}s", colors=["GRAY", "ORANGE"])
        logger.log(
            f"Total Execution Time:", f"{(search_execution_time + rerank_execution_time):.2f}s", colors=["GRAY", "ORANGE"])
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        model.close()
