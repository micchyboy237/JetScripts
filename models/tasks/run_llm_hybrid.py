import os
import time
from llama_cpp import Llama

from jet.data.utils import generate_key
from jet.file.utils import load_file, save_file
from jet.models.tasks.llm_search import search_docs
from jet.models.tasks.llm_rerank import rerank_docs
from jet.models.tokenizer.base import count_tokens, get_tokenizer

if __name__ == "__main__":
    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    query = "List all ongoing and upcoming isekai anime 2025."
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        query
    ]
    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            # doc["metadata"]["content"],
        ]).strip()
        for doc in docs
    ]
    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/8aa0010e73a1075e99dfc213a475a60fd971bbe7/Qwen3-Embedding-0.6B-f16.gguf"
    tokenizer_name = "mlx-community/Qwen3-0.6B-4bit"

    tokenizer = get_tokenizer(tokenizer_name)
    token_counts: list[int] = count_tokens(
        tokenizer, documents, prevent_total=True)
    largest_size = max(token_counts)

    print("Largest doc tokens:", largest_size)

    settings = {
        "model_path": model_path,
        "embedding": True,
        "n_ctx": largest_size + 32,
        "n_threads": 4,
        "n_gpu_layers": -1,
        "n_threads_batch": 64,
        "no_perf": True,      # Disable performance timings
        "verbose": True,
        "flash_attn": True,
    }
    key = generate_key(**settings)
    model = Llama(**settings)

    output_key_dir = os.path.join(output_dir, key)

    try:
        # Initial search
        print("Starting search docs...")
        search_start_time = time.time()
        search_results = search_docs(model, queries, documents, task)
        for query_idx, query_results in enumerate(search_results):
            print(f"\nQuery: {queries[query_idx]} (Search Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
        search_execution_time = time.time() - search_start_time
        save_file({
            "execution_time": f"{search_execution_time:.2f}s",
            "query": query,
            "count": len(search_results),
            "results": search_results
        }, f"{output_key_dir}/search_results.json")

        # Rerank results
        print("Starting rerank docs...")
        rerank_start_time = time.time()
        rerank_results = rerank_docs(model, queries, search_results)
        for query_idx, query_results in enumerate(rerank_results):
            print(f"\nQuery: {queries[query_idx]} (Reranked Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
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
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        model.close()

        print("\nExecution Times:")
        logger.log(f"Search Execution Time:",
                   f"{search_execution_time:.2f}s", colors=["GRAY", "ORANGE"])
        logger.log(f"Rerank Execution Time:",
                   f"{rerank_execution_time:.2f}s", colors=["GRAY", "ORANGE"])
        logger.log(
            f"Total Execution Time:", f"{(search_execution_time + rerank_execution_time):.2f}s", colors=["GRAY", "ORANGE"])
