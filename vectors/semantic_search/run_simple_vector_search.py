import os
import shutil

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data_with_info
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # model_name: OLLAMA_MODEL_NAMES = "embeddinggemma"
    model_name: OLLAMA_MODEL_NAMES = "nomic-embed-text"
    # Same example queries
    query = "Top isekai anime 2025"
    chunks_with_info = load_sample_data_with_info(model=model_name, includes=["p"])
    save_file(chunks_with_info, f"{OUTPUT_DIR}/chunks_with_info.json")

    documents = [chunk["content"] for chunk in chunks_with_info]
    token_counts = [chunk["num_tokens"] for chunk in chunks_with_info]
    logger.debug(f"Total tokens: {sum(token_counts)}")

    save_file(
        {
            "count": len(documents),
            "tokens": {
                "max": max(token_counts),
                "min": min(token_counts),
                "total": sum(token_counts),
            },
            "results": [
                {"doc_index": idx, "tokens": tokens, "text": text}
                for idx, (tokens, text) in enumerate(zip(token_counts, documents))
            ],
        },
        f"{OUTPUT_DIR}/documents.json",
    )

    search_engine = VectorSearch(model_name)

    search_results = search_engine.search(query, documents)
    print(f"\nQuery: {query}")
    print("Top matches:")
    for result in search_results[:10]:
        print(
            f"\n{colorize_log(f'{result["rank"]}.', 'ORANGE')} (Score: "
            f"{colorize_log(f'{result["score"]:.3f}', 'SUCCESS')})"
        )
        print(f"{result['text']}")

    save_file(
        {"query": query, "count": len(search_results), "results": search_results},
        f"{OUTPUT_DIR}/search_results.json",
    )
