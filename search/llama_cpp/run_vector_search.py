import os
import shutil
from jet.vectors.semantic_search.search_docs import search_docs
from jet.file.utils import load_file, save_file
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main(query: str, md_content: str, chunk_size: int, chunk_overlap: int, model: str = "embeddinggemma", threshold: float = 0.0):
    """Main function to process markdown content, chunk it, and perform search with optional threshold."""
    chunks = chunk_texts_with_data(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
    print(f"Number of chunks: {len(chunks)}")
    save_file(chunks, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/chunks.json")
    
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    
    search_results = search_docs(query, texts, model, ids=ids, threshold=threshold)
    
    # Add token count from original chunk
    for result in search_results:
        result["tokens"] = chunks[result["doc_index"]]["num_tokens"]
    
    return search_results

if __name__ == '__main__':
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/markdown_no_links.md"
    md_content: str = load_file(md_file)
    save_file(md_content, f"{OUTPUT_DIR}/doc.md")

    query = "How to change max depth?"
    model = "embeddinggemma"

    # Test various chunk sizes, overlaps, and thresholds
    chunk_sizes = [128, 64, 32]
    chunk_overlaps = [32, 16]
    threshold = 0.0

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            print(f"\n--- Searching with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, threshold={threshold} ---")
            search_results = main(query, md_content, chunk_size, chunk_overlap, model, threshold=threshold)
            save_file({
                "model": model,
                "query": query,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "threshold": threshold,
                "count": len(search_results),
                "results": search_results,
            }, f"{OUTPUT_DIR}/chunked_{chunk_size}_{chunk_overlap}/search_results.json")
