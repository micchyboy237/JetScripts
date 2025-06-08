# Example usage
import os
import time
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.tasks.llm_rerank import rerank_docs
from jet.models.tokenizer.utils import calculate_n_ctx


if __name__ == "__main__":
    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    query = "List all ongoing and upcoming isekai anime 2025."
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            # doc["metadata"]["content"],
        ]).strip()
        for doc in docs
    ]
    doc_ids = [doc["id"] for doc in docs]

    # task = 'Given a web search query, retrieve relevant passages that answer the query'
    # queries = ["What is the capital of China?", "Explain gravity"]
    # documents = [
    #     "The capital of China is Beijing.",
    #     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    # ]
    # doc_ids = ["doc1", "doc2"]

    logger.info("\nStarting rerank docs...")
    rerank_start_time = time.time()
    rerank_results = rerank_docs(query, documents, task,
                                 ids=doc_ids, show_progress=True)
    rerank_execution_time = time.time() - rerank_start_time

    logger.log(f"Execution time:",
               f"{rerank_execution_time:.2f}s", colors=["GRAY", "ORANGE"])

    save_file({
        "execution_time": f"{rerank_execution_time:.2f}s",
        "query": query,
        "count": len(rerank_results),
        "results": rerank_results,
    }, f"{output_dir}/results.json")
