import os
from typing import List
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.file_vector_search import FileSearchResult, search_files


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def save_results(query: str, directories: List[str], results: List[FileSearchResult], split_chunks: bool):
    print(f"Search results for '{query}' in these dirs:")
    for d in directories:
        print(d)
    for num, result in enumerate(results, start=1):
        file_path = result["metadata"]["file_path"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nFile: {file_path}")

    save_file({
        "query": query,
        "count": len(results),
        "merged": not split_chunks,
        "results": results
    }, f"{OUTPUT_DIR}/results_{'split' if split_chunks else 'merged'}.json")


def main():
    """Main function to demonstrate file search."""
    # Example usage
    directories = [
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_notes",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors",
    ]

    query = "BM25 reranking"
    extensions = [".py"]
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"

    top_k = 10
    threshold = 0.0  # Using default threshold
    chunk_size = 500
    chunk_overlap = 100

    split_chunks = True
    with_split_chunks_results = list(
        search_files(
            directories, query, extensions,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks
        )
    )
    save_results(query, directories, with_split_chunks_results, split_chunks)

    split_chunks = False
    without_split_chunks_results = list(
        search_files(
            directories, query, extensions,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks
        )
    )
    save_results(query, directories,
                 without_split_chunks_results, split_chunks)


if __name__ == "__main__":
    main()
