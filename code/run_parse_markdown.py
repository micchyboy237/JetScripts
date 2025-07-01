import os
import shutil

from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/bakabuzz_com_26_upcoming_isekai_anime_of_2025_you_must_watch/page.html"

    results_with_links = parse_markdown(html_file, ignore_links=False)
    results_no_links = parse_markdown(html_file, ignore_links=True)

    save_file(results_with_links, f"{output_dir}/results_with_links.json")
    save_file(results_no_links, f"{output_dir}/results_no_links.json")
