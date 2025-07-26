import os
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.vectors.semantic_search.file_vector_search import search_files


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def main():
    """Main function to demonstrate file search."""
    # Example usage
    directories = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_notes",
    ]
    query = "test file"
    extensions = [".py"]
    top_k = 10
    threshold = 0.0  # Using default threshold

    results = list(search_files(directories, query, extensions,
                   top_k=top_k, threshold=threshold))
    print(f"Search results for '{query}' in these dirs:")
    for d in directories:
        print(d)
    for num, result in enumerate(results, start=1):
        file_path = result["metadata"]["file_path"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Start - End: {start_idx} - {end_idx}\nFile: {file_path}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
