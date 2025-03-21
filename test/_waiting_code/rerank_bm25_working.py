from jet.file.utils import load_file
from jet.utils.commands import copy_to_clipboard
from jet.logger import logger
from jet.vectors.reranker.bm25_helpers import search_and_rerank


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    texts: list[str] = load_file(data_file)
    queries = ["Season", "episode", "synopsis"]

    search_results = search_and_rerank(queries, texts)

    copy_to_clipboard(search_results)

    for idx, result in enumerate(search_results["data"][:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
