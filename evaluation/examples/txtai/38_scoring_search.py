from typing import get_args
import json
import os
import traceback
from jet.libs.txtai.vectors import VectorsFactory
from jet.llm.search import (
    ScoringMethod,
    load_local_json,
    load_or_create_embeddings,
    build_ann_index,
    ann_search,
    scoring_search,
)
from jet.logger import logger

GENERATED_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/evaluation/examples/txtai/generated"
# Cache directory for embeddings
EMBEDDINGS_DIR = f"{GENERATED_DIR}/embeddings"
RESULTS_DIR = f"{GENERATED_DIR}/search"
EMBEDDINGS_CACHE_KEY = "crew_ai_docs"


def save_results(results, filename):
    """Save results to a JSON file."""
    results_path = os.path.join(RESULTS_DIR, filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.log("Saved", f"({len(results['results'])})", "to:", results_path, colors=[
               "LOG", "DEBUG", "LOG", "BRIGHT_SUCCESS"])


def main():
    embedding_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    dataset_path = "/Users/jethroestrada/Desktop/External_Projects/AI/agents_2/crewAI/my_project/src/my_project/generated/rag/crewai-docs.json"
    dataset = load_local_json(dataset_path)
    texts = [row["page_content"] for row in dataset]
    embeddings_model = VectorsFactory.create({"path": embedding_model}, None)
    embeddings_dir = os.path.join(EMBEDDINGS_DIR, EMBEDDINGS_CACHE_KEY)
    cache_file = os.path.join(embeddings_dir, "embeddings.npy")
    embeddings = load_or_create_embeddings(
        texts, model=embeddings_model, cache_file=cache_file)

    query = "crewai setup"
    top_k = 5

    # Perform ANN search
    ann = build_ann_index(embeddings)
    ann_results = ann_search(
        query, ann, model=embeddings_model, dataset=dataset, top_k=top_k)

    save_results(
        {
            "type": "ann",
            "embedding_model": embedding_model,
            "query": query,
            "top_k": top_k,
            "results": ann_results
        },
        "ann_scores.json")

    # Filter the dataset for scoring search
    ann_ids = {result["id"] for result in ann_results}
    filtered_dataset = dataset.filter(lambda example: example["id"] in ann_ids)

    # Perform scoring search for each method
    scoring_methods: list[ScoringMethod] = list(get_args(ScoringMethod))
    for method in scoring_methods:
        try:
            scoring_results = scoring_search(
                query, dataset=dataset, method=method, top_k=top_k)

            save_results({
                "type": "vector",
                "method": method,
                "query": query,
                "top_k": top_k,
                "results": scoring_results
            }, f"vector_{method}_scores.json")
        except Exception as e:
            logger.error(e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
