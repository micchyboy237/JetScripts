import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # dimensions = 512
    # model_name: OLLAMA_MODEL_NAMES = "mxbai-embed-large"
    # model_name: OLLAMA_MODEL_NAMES = "nomic-embed-text"
    # model_name: OLLAMA_MODEL_NAMES = "all-MiniLM-L6-v2"
    model_name: OLLAMA_MODEL_NAMES = "embeddinggemma"
    # model_name: OLLAMA_MODEL_NAMES = "static-retrieval-mrl-en-v1"
    # Same example queries
    queries = [
        "How to change max depth?",
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    # Same sample documents
    sample_docs = [
        "##### Help\n\n- Help Center",
        "##### Legal\n\n- Security & Compliance\n- Privacy Policy",
        "##### Partnerships\n\n- IBM",
        "Find all pages about the Python SDK\" `\nmax_depth\ninteger\ndefault: 1\nMax depth of the crawl. Defines how far from the base URL the crawler can explore.\nRequired range: ` x >= 1 `\nmax_breadth\ninteger\ndefault: 20",
        "How to change max depth?",
    ]
    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
