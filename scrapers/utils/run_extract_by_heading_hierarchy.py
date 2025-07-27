import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_rag_strategies_reddit_2025/pages/medium_com_aa779_rag_in_2025_7_proven_strategies_to_deploy_retrieval_augmented_generation_at_scale_d1f71dfbfbba/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    header_elements = extract_by_heading_hierarchy(html_str)
    save_file(header_elements, f"{output_dir}/headings.json")
