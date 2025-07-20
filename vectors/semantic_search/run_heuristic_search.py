import os
import shutil
from jet.file.utils import load_file, save_file
from jet.vectors.reranker.bm25 import get_bm25_similarities, get_bm25_similarities_old
from jet.wordnet.keywords.helpers import extract_query_candidates
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)
    texts = [f"{d["title"]}\n{d["details"]}" for d in data]

    # query = "React Native"
    query = "React.js web with Python AI development"

    query_candidates = extract_query_candidates(query)
    results = get_bm25_similarities(query_candidates, texts)

    save_file({
        "query": query,
        "query_candidates": query_candidates,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")

    results_old = get_bm25_similarities_old(query_candidates, texts)

    save_file({
        "query": query,
        "query_candidates": query_candidates,
        "count": len(results_old),
        "results": results_old
    }, f"{OUTPUT_DIR}/results_old.json")
