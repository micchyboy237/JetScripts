
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
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_notes",
    ]
    query = "test file"
    extensions = {".py", ".txt"}
    top_k = 10

    results = search_files(directories, query, extensions, top_k=top_k)
    print(f"Search results for '{query}' in these dirs:")
    for d in directories:
        print(d)
    for file_path, score in results:
        print(
            f"Score: {colorize_log(f"{score:.3f}", "SUCCESS")} | File: {file_path}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
