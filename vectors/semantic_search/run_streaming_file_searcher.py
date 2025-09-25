import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.vectors.semantic_search.streaming_file_searcher import FileSearcher

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    searcher = FileSearcher(
        base_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/langchain",
        threshold=0.0,
        includes=["*.py"],
    )
    query = "task completion agents"
    count = 0
    results = []
    for file_path, score in searcher.search(query):
        count += 1
        results.append({
            "score": score,
            "file": file_path
        })
        # Sort results by score descending and add rank
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(results_sorted, 1):
            item["rank"] = idx
        results[:] = results_sorted
        print(f"\n{colorize_log(f"{count}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
        print(file_path)

        save_file({
            "query": query,
            "count": len(results),
            "results": results
        }, f"{OUTPUT_DIR}/results.json", verbose=count == 0)
