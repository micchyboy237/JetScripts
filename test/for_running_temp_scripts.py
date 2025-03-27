import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from typing import List, Dict, Callable, Optional, TypedDict
from jet.file.utils import load_file, save_file
from jet.llm.utils.embeddings import get_embedding_function
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import cluster_texts, get_query_similarity_scores
from jet.logger import logger
from jet.vectors.reranker.bm25_helpers import HybridSearch

queries_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data/aniwatch_history.json"
queries: list[str] = load_file(queries_path)

db_path = DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data/anime.db"
table_name = "jet_history"


def load_all_records(db_path: str, table_name: str):
    with sqlite3.connect(db_path, timeout=10) as conn:
        cursor = conn.cursor()

        # Fetch all records
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names from cursor description
        columns = [col[0] for col in cursor.description]

        # Convert each row to dictionary
        results = [dict(zip(columns, row)) for row in rows]

    return results


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]
    title: str
    url: str
    image_url: str
    score: Optional[float]
    episodes: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    status: str
    members: Optional[int]
    anime_type: Optional[str]


data: list[ScrapedData] = load_all_records(db_path, table_name)
data_dict: dict[str, ScrapedData] = {d["id"]: d for d in data}
ids = [d["id"] for d in data]
texts = [d["title"] for d in data]

embed_model = "nomic-embed-text"
hybrid_search = HybridSearch(model_name=embed_model)
hybrid_search.build_index(texts, ids=ids)

top_k = None
threshold = 0.0


if __name__ == "__main__":
    while True:
        query = input("Enter query (or 'q' to quit): ")
        # quit on 'q' or ctrl+c
        if query == 'q' or query == KeyboardInterrupt:
            logger.info("Exiting...")
            break

        results = hybrid_search.search(query, top_k=top_k, threshold=threshold)
        results = results.copy()

        semantic_results = results.pop("semantic_results")
        hybrid_results = results.pop("hybrid_results")
        reranked_results = results.pop("reranked_results")
        reranked_data = [data_dict[result["id"]]
                         for result in reranked_results]

        logger.debug(f"Query: {query} | Results: {len(reranked_results)}")
        for idx, result in enumerate(reranked_results):
            logger.log(f"[{idx + 1}]", f"{result["score"]:.2f}",
                       result["text"][:30], colors=["INFO", "SUCCESS", "WHITE"])
            logger.gray(reranked_data[idx]["url"])

        save_file(results, "generated/hybrid_search/results_info.json")
        save_file({"query": query, "results": semantic_results},
                  "generated/hybrid_search/semantic_results.json")
        save_file({"query": query, "results": hybrid_results},
                  "generated/hybrid_search/hybrid_results.json")
        save_file({"query": query, "results": reranked_results},
                  "generated/hybrid_search/reranked_results.json")
        save_file({"query": query, "results": reranked_data},
                  "generated/hybrid_search/reranked_data.json")
