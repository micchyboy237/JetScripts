import os
import shutil
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.code.markdown_utils._base import read_html_content
from jet.data.header_docs import HeaderDocs
from jet.data.header_utils import split_and_merge_headers, prepare_for_rag, search_headers
from jet.file.utils import load_file, save_file
from jet.logger import logger


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/bakabuzz_com_26_upcoming_isekai_anime_of_2025_you_must_watch/page.html"
    # html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    page_with_links = read_html_content(html_file, ignore_links=False)
    page_no_links = read_html_content(html_file, ignore_links=True)

    save_file(page_with_links, f"{output_dir}/page_with_links.html")
    save_file(page_no_links, f"{output_dir}/page_no_links.html")
