import os
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.keywords.keyword_search_spellchecker import KeywordVectorSearchSpellChecker, SearchResult
from typing import List

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    words = ["the", "quick", "brown", "fox", "jumps"]
    searcher = KeywordVectorSearchSpellChecker()
    searcher.build_index(words)
    query = "teh"
    results: List[SearchResult] = searcher.search(query, k=3)
    logger.success(format_json(results))
    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/example/results.json")
