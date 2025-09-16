import os
import shutil
from jet.file.utils import load_file, save_file
from jet.vectors.reranker.bm25 import get_bm25_similarities, get_bm25_similarities_old
from jet.wordnet.keywords.helpers import extract_query_candidates
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def main(query: str):
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)
    texts = [f"{d["title"]}\n{d["details"]}" for d in data]

    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(output_dir, ignore_errors=True)

    query_candidates = extract_query_candidates(query)
    ids = [job["id"] for job in data]

    results = get_bm25_similarities(query_candidates, texts, ids=ids)

    save_file({
        "query": query,
        "query_candidates": query_candidates,
        "count": len(results),
        "results": results
    }, f"{output_dir}/results.json")

    results_old = get_bm25_similarities_old(query_candidates, texts, ids=ids)

    save_file({
        "query": query,
        "query_candidates": query_candidates,
        "count": len(results_old),
        "results": results_old
    }, f"{output_dir}/results_old.json")


if __name__ == "__main__":
    query = "React Native"
    main(query)
    query = "React.js web with Python AI development"
    main(query)
