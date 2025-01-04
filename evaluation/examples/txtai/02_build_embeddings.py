import json
import os
from datasets import Dataset
from txtai.pipeline import Similarity
from jet.llm.embeddings import load_or_create_embeddings
from jet.logger import logger

# Cache directory for embeddings
CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/evaluation/examples/embeddings_cache"
EMBEDDINGS_CACHE_KEY = "crew_ai_docs"


def load_local_json(filepath):
    """
    Load a JSON file into a Hugging Face dataset.

    Args:
    - filepath: Path to the local JSON file.

    Returns:
    - A Hugging Face Dataset object.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return Dataset.from_dict({"id": [item["id"] for item in data],
                              "tags": [item["metadata"]["tags"] for item in data],
                              "page_content": [item["page_content"] for item in data]})


def stream(dataset, field, limit):
    """
    Streams data from a dataset, yielding tuples of index, text, and None until the limit is reached.
    """
    index = 0
    for row in dataset:
        yield (index, row[field], None)
        index += 1
        if index >= limit:
            break


def search(query):
    """
    Searches the embeddings index for a given query and returns results.
    """
    return [(result["score"], result["text"]) for result in embeddings.search(query, 50)]


def ranksearch(query):
    """
    Re-ranks search results for a query using the similarity model.
    """
    results = [text for _, text in search(query)]
    return [(score, results[x]) for x, score in similarity(query, results)]


def table(query, rows):
    """
    Displays a table of search results in plain text.
    """
    print(f"\n{query}")
    print(f"{'Score':<10}{'Text'}")
    for score, text in rows:
        print(f"{score:<10.4f}{text}")


def main():
    """
    Main function to load a dataset, build embeddings, and run search queries.
    """
    embedding_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    dataset_path = "/Users/jethroestrada/Desktop/External_Projects/AI/agents_2/crewAI/my_project/src/my_project/generated/rag/crewai-docs.json"

    # Load local JSON dataset
    dataset = load_local_json(dataset_path)

    # Load or create embeddings (with caching)
    global embeddings
    cache_file = os.path.join(CACHE_DIR, EMBEDDINGS_CACHE_KEY)
    embeddings = load_or_create_embeddings(
        dataset, embedding_model, cache_file)

    # Create similarity instance for re-ranking
    global similarity
    similarity = Similarity("valhalla/distilbart-mnli-12-3")

    # Run and display queries
    queries = [
        "How to setup and use crew AI features",
        "Crew AI flows",
    ]
    for query in queries:
        table(query, ranksearch(query)[:2])


if __name__ == "__main__":
    main()
