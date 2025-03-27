

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from typing import List, Dict, Callable, Optional, TypedDict
from jet.file.utils import load_file
from jet.llm.utils.embeddings import get_embedding_function
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import cluster_texts
from jet.logger import logger
from jet.actions.vector_semantic_search import VectorSemanticSearch

queries_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data/aniwatch_history.json"


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

doc_texts = [d["title"] for d in data]


# Define queries and candidates
# queries = [
#     "Used technologies are: Magento, front-end architecture, HTML, CSS, JavaScript, front-end technologies, jQuery, Git, React, Vue.js"
# ]
# candidates = [
#     "React.js",
#     "React Native",
#     "Node.js",
#     "Python",
#     "PostgreSQL",
#     "MongoDB",
#     "Firebase",
#     "AWS",
# ]
queries: list[str] = load_file(queries_path)[:3]
candidates = doc_texts


if __name__ == "__main__":
    search = VectorSemanticSearch(candidates)

    # # Perform Vector-based search
    # results = search.vector_based_search(queries)
    # logger.newline()
    # logger.orange(f"Vector-Based Search Results ({len(results)}):")
    # for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
    #     logger.newline()
    #     logger.log(" -", f"Query {query_idx}:",
    #                query_line, colors=["GRAY", "GRAY", "DEBUG"])
    #     for result in group[:5]:
    #         logger.log("  +", f"{result['text'][:25]}:", f"{
    #                    result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # # Perform Fusion search
    # results = search.fusion_search(queries)
    # logger.newline()
    # logger.orange(f"Fusion Search Results ({len(results)}):")
    # for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
    #     logger.newline()
    #     logger.log(" -", f"Query {query_idx}:",
    #                query_line, colors=["GRAY", "GRAY", "DEBUG"])
    #     for result in group[:5]:
    #         logger.log("  +", f"{result['text'][:25]}:", f"{
    #                    result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # # Perform FAISS search
    # results = search.faiss_search(queries)
    # logger.newline()
    # logger.orange(f"FAISS Search Results ({len(results)}):")
    # for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
    #     logger.newline()
    #     logger.log(" -", f"Query {query_idx}:",
    #                query_line, colors=["GRAY", "GRAY", "DEBUG"])
    #     for result in group[:5]:
    #         logger.log("  +", f"{result['text'][:25]}:", f"{
    #                    result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # # Perform Graph-based search
    # results = search.graph_based_search(queries)
    # logger.newline()
    # logger.orange("Graph-Based Search Results:")
    # for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
    #     logger.newline()
    #     logger.log(" -", f"Query {query_idx}:",
    #                query_line, colors=["GRAY", "GRAY", "DEBUG"])
    #     for result in group[:5]:
    #         logger.log("  +", f"{result['text'][:25]}:", f"{
    #                    result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # # Perform Annoy search
    # results = search.annoy_search(queries)
    # logger.newline()
    # logger.orange(f"Annoy Search Results ({len(results)}):")
    # for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
    #     logger.newline()
    #     logger.log(" -", f"Query {query_idx}:",
    #                query_line, colors=["GRAY", "GRAY", "DEBUG"])
    #     for result in group[:5]:
    #         logger.log("  +", f"{result['text'][:25]}:", f"{
    #                    result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform BM25 search
    results = search.bm25_search(queries)
    logger.newline()
    logger.orange(f"BM25 Search Results ({len(results)}):")
    for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group[:5]:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Cross-encoder search
    results = search.cross_encoder_search(queries)
    logger.newline()
    logger.orange("Cross-Encoder Search Results:")
    for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group[:5]:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Rerank search
    results = search.rerank_search(queries)
    logger.newline()
    logger.orange("Rerank Search Results:")
    for query_idx, (query_line, group) in enumerate(list(results.items())[:3]):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group[:5]:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])
