import os
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.keywords.mispelled_keyword_vector_search import MispelledKeywordVectorSearch, SearchResult
from typing import List

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    # Initialize searcher with custom word list
    words = ["the", "quick", "brown", "fox", "jumps"]
    searcher = MispelledKeywordVectorSearch()
    searcher.build_index(words)

    # Test with single string
    query = "teh quik foxx jumpss"
    results: List[SearchResult] = searcher.search(query, k=5)
    logger.success(format_json(results))
    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/example/single_string_results.json")

    # Test with list of documents
    documents = [
        "teh quick brown foxx",
        "jummps over teh lazy dogg"
    ]
    results: List[SearchResult] = searcher.search(documents, k=5)
    logger.success(format_json(results))
    save_file({
        "query": documents,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/example/document_list_results.json")
