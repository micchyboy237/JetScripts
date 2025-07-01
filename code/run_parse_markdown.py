import os

from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/bakabuzz_com_26_upcoming_isekai_anime_of_2025_you_must_watch/page.html"

    html_file = '\n# Main Header\nMain content\n\n## Sub Header\nSub content\n\nParagraph content\n'

    results = parse_markdown(
        html_file, merge_headers=False, merge_contents=True)

    results2 = parse_markdown(html_file)

    save_file(results, f"{output_dir}/results.json")
