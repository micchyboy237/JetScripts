from jet.wordnet.words import count_words
import numpy as np
import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from typing import List, Dict, Optional, TypedDict
from jet.file.utils import load_file, save_file
from jet.llm.utils.embeddings import get_embedding_function
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import cluster_texts, get_query_similarity_scores
from jet.logger import logger
from jet.vectors.reranker.bm25_helpers import HybridSearch

# File Paths
db_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data/anime.db"
table_name = "jet_history"

embed_model = "nomic-embed-text"


DB_CONFIG = {
    "dbname": "anime_db1",
    "user": "jethroestrada",
    "password": "",
    "host": "jetairm1",
    "port": "5432"
}
TABLE_NAME = "history"

# Load database records


def load_all_records():
    # with sqlite3.connect(db_path, timeout=10) as conn:
    #     cursor = conn.cursor()
    #     cursor.execute(f"SELECT * FROM {table_name}")
    #     rows = cursor.fetchall()
    #     columns = [col[0] for col in cursor.description]
    # return [dict(zip(columns, row)) for row in rows]
    conn = psycopg.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        autocommit=False,  # Enable manual transaction control
        row_factory=dict_row
    )

    query = f"SELECT * FROM {TABLE_NAME};"
    with conn.cursor() as cur:
        cur.execute(query)
        results = cur.fetchall()
        return results


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]
    title: Optional[str]
    url: Optional[str]
    image_url: Optional[str]
    score: Optional[float]
    episodes: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    next_date: Optional[str]
    status: Optional[str]
    members: Optional[int]
    anime_type: Optional[str]
    average_score: Optional[int]
    mean_score: Optional[int]
    favorites: Optional[int]
    next_episode: Optional[int]
    popularity: Optional[int]
    demographic: Optional[str]
    studios: Optional[str]
    producers: Optional[str]
    source: Optional[str]
    japanese: Optional[str]
    english: Optional[str]
    synonyms: Optional[str]
    tags: Optional[str]
    synopsis: Optional[str]
    genres: Optional[str]


# Prepare data
data: list[ScrapedData] = load_all_records()

# Get all unique genres
unique_genres = set()
for d in data:
    if d["genres"]:
        # Split genres string and add each genre to set
        genres = d["genres"].split(",")
        unique_genres.update(g.strip() for g in genres)
unique_genres = sorted(list(unique_genres))

# Get all unique tags
unique_tags = set()
for d in data:
    if d["tags"]:
        # Split tags string and add each tag to set
        tags = d["tags"].split(",")
        unique_tags.update(g.strip() for g in tags)
unique_tags = sorted(list(unique_tags))

data_dict: dict[str, ScrapedData] = {d["id"]: d for d in data}
ids = [d["id"] for d in data]


queries = [d["english"] or d["title"]
           for d in data if count_words(d["english"] or d["title"]) <= 3]


# Search sample
if __name__ == "__main__":
    texts = []
    for d in data:
        title = f"Title: {d.get('english') or d.get('title')}"
        synopsis = f"Synopsis: {d.get('synopsis')}"
        synonyms = f"Synonyms: {d.get('synonyms')}"
        tags_str = d.get('tags', '')
        tags = [f"Tag: {tag.strip()}" for tag in tags_str.split(',')
                if tag.strip()] if tags_str else []
        genres_str = d.get('genres', '')
        genres = [f"Genre: {genre.strip()}" for genre in genres_str.split(
            ',') if genre.strip()] if genres_str else []

        text_parts = [title]
        if d.get('synopsis'):
            text_parts.append(synopsis)
        if d.get('synonyms'):
            text_parts.append(synonyms)
        if tags:
            text_parts.extend(tags)
        if genres:
            text_parts.extend(genres)
        text = "\n".join(text_parts)

        texts.append(text)

    # Store results
    search_results = []

    # Sort queries by length
    queries = sorted(queries, key=len)
    queries = queries[:10]

    # Initialize Hybrid Search
    hybrid_search = HybridSearch(model_name=embed_model)
    hybrid_search.build_index(texts, ids=ids)

    # Search Config
    top_k = None
    threshold = 0.0

    for query in tqdm(queries):
        query = f"Title: {query}"
        results = hybrid_search.search(query, top_k=top_k, threshold=threshold)
        semantic_results = results.pop("semantic_results")
        hybrid_results = results.pop("hybrid_results")
        reranked_results = results.pop("reranked_results")
        result = [{
            "query": query,
            "semantic_results": [{
                "id": result["id"],
                "score": result["score"],
                "similarity": result.get("similarity"),
                "text": result["text"],
                "matched": result["matched"],
            } for result in semantic_results],
            "reranked_results": [{
                "id": result["id"],
                "score": result["score"],
                "similarity": result.get("similarity"),
                "text": result["text"],
                "matched": result["matched"],
            } for result in reranked_results],
            "hybrid_results": [{
                "id": result["id"],
                "score": result["score"],
                "similarity": result.get("similarity"),
                "text": result["text"],
                "matched": result["matched"],
            } for result in hybrid_results],
            # "data": [data_dict[result["id"]] for result in reranked_results],
        }]

        logger.debug(f"Query: {query} | Results: {len(reranked_results)}")

        search_results.append(result)

        save_file(search_results, "generated/hybrid_search/search_results.json")

    logger.info("All queries processed and results saved.")

# Cluster similar texts
if __name__ == "__main__":
    texts = []
    for d in data:
        title = f"Title: {d.get('english') or d.get('title')}"
        synopsis = f"Synopsis: {d.get('synopsis')}"
        synonyms = f"Synonyms: {d.get('synonyms')}"
        tags_str = d.get('tags', '')
        tags = [f"Tag: {tag.strip()}" for tag in tags_str.split(',')
                if tag.strip()] if tags_str else []
        genres_str = d.get('genres', '')
        genres = [f"Genre: {genre.strip()}" for genre in genres_str.split(
            ',') if genre.strip()] if genres_str else []

        text_parts = [title]
        if d.get('synopsis'):
            text_parts.append(synopsis)
        if d.get('synonyms'):
            text_parts.append(synonyms)
        if tags:
            text_parts.extend(tags)
        if genres:
            text_parts.extend(genres)
        text = "\n".join(text_parts)

        texts.append(text)

    embed_func = get_embedding_function(embed_model)
    clustered_texts = cluster_texts(texts, embed_func)

    save_file(clustered_texts, "generated/anime_clustered_texts.json")


# if __name__ == "__main__":
#     while True:
#         query = input("Enter query (or 'q' to quit): ")
#         # quit on 'q' or ctrl+c
#         if query == 'q' or query == KeyboardInterrupt:
#             logger.info("Exiting...")
#             break

#         results = hybrid_search.search(query, top_k=top_k, threshold=threshold)
#         results = results.copy()

#         semantic_results = results.pop("semantic_results")
#         hybrid_results = results.pop("hybrid_results")
#         reranked_results = results.pop("reranked_results")
#         reranked_data = [data_dict[result["id"]]
#                          for result in reranked_results]

#         logger.debug(f"Query: {query} | Results: {len(reranked_results)}")
#         for idx, result in enumerate(reranked_results):
#             logger.log(f"[{idx + 1}]", f"{result["score"]:.2f}",
#                        result["text"][:30], colors=["INFO", "SUCCESS", "WHITE"])
#             logger.gray(reranked_data[idx]["url"])

#         save_file(results, "generated/hybrid_search/results_info.json")
#         save_file({"query": query, "results": semantic_results},
#                   "generated/hybrid_search/semantic_results.json")
#         save_file({"query": query, "results": hybrid_results},
#                   "generated/hybrid_search/hybrid_results.json")
#         save_file({"query": query, "results": reranked_results},
#                   "generated/hybrid_search/reranked_results.json")
#         save_file({"query": query, "results": reranked_data},
#                   "generated/hybrid_search/reranked_data.json")
