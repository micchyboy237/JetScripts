import os
import shutil
import asyncio
import argparse
from jet.file.utils import save_file
from jet.search.deep_search import web_deep_search
from jet.utils.text import format_sub_dir

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

    sub_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(sub_dir, ignore_errors=True)

    result = asyncio.run(web_deep_search(query))

    save_file(f"""\
## Prompt

### Query

{result["query"]}

### Template

{result["template"]}

### Context

{result["context"]}

## Response

{result["response"]}
""", f"{sub_dir}/prompt_response.md")

    save_file(result["search_engine_results"], f"{sub_dir}/search_engine_results.json")
    save_file(result["started_urls"], f"{sub_dir}/started_urls.json")
    save_file(result["searched_urls"], f"{sub_dir}/searched_urls.json")
    save_file(result["high_score_urls"], f"{sub_dir}/high_score_urls.json")
    save_file(result["header_docs"], f"{sub_dir}/header_docs.json")
    save_file(result["search_results"], f"{sub_dir}/search_results.json")
    save_file(result["sorted_search_results"], f"{sub_dir}/sorted_search_results.json")
    save_file(result["filtered_results"], f"{sub_dir}/filtered_results.json")
    save_file(result["filtered_urls"], f"{sub_dir}/filtered_urls.json")
    save_file(result["token_info"], f"{sub_dir}/token_info.json")
