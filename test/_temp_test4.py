
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.tasks.evaluate_relevance import evaluate_relevance
from jet.models.tasks.search_docs import search_docs


if __name__ == "__main__":
    import os
    from jet.file.utils import load_file, save_file

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            # doc["metadata"]["content"]
        ]).strip()
        for doc in docs
    ]
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    query = "List all ongoing and upcoming isekai anime 2025."
    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    # input_ids_list = generate_embeddings(
    #     documents, model_name, show_progress=True)
    # logger.success(f"Done embedding documents ({len(input_ids_list)})")
    # save_file(input_ids_list, f"{output_dir}/embeddings.json")

    results = evaluate_relevance(
        query, documents, task, model_name, show_progress=True)
    save_file({
        "task": task,
        "query": query,
        "results": results
    }, f"{output_dir}/results.json")

    # # Initialize tagger
    # results = search_docs(query, documents, task)
