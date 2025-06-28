import os

from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes.in_15_best_upcoming_isekai_anime_in_2025/page.html"
    # html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"

    results = parse_markdown(
        html_file, merge_headers=False, merge_contents=True)

    save_file(results, f"{output_dir}/results.json")
