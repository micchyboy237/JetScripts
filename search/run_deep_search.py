import os
import shutil
import asyncio
import argparse
from jet.file.utils import save_file
from jet.search.deep_search import web_deep_search

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("query_pos", type=str, nargs="?",
                   help="Search query as positional argument")
    p.add_argument("-q", "--query", type=str,
                   help="Search query using optional flag")
    args = p.parse_args()

    query = args.query if args.query else args.query_pos or "Top 10 isekai anime 2025 with release date, synopsis, number of episode, airing status"

    deep_search_result = asyncio.run(web_deep_search(query))
    save_file(deep_search_result, f"{OUTPUT_DIR}/deep_search_result.json")