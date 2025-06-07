import os
import time
from llama_cpp import Llama
import llama_cpp

from jet.file.utils import load_file, save_file
from jet.models.tasks.llm_search import search_docs


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

    model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
    model = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=-1,
        n_threads_batch=8,
        no_perf=True,      # Disable performance timings
        verbose=True
    )

    try:
        print("Starting search docs...")
        start_time = time.time()
        results = search_docs(model, queries, documents, task)
        execution_time = time.time() - start_time
        print(f"Completed search_docs in {execution_time:.2f}s")
        for query_idx, query_results in enumerate(results[:10]):
            print(f"\nQuery: {queries[query_idx]}")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
        save_file({
            "execution_time": f"{execution_time:.2f}s",
            "query": query,
            "count": len(results),
            "results": results
        }, f"{output_dir}/search_results.json")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        model.close()
