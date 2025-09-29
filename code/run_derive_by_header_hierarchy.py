import os

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/async_results/html_files/https_cloud_google_com_blog_topics_public_sector_5_ai_trends_shaping_the_future_of_the_public_sector_in_2025.html"
    html_str: str = load_file(html_file)
    md_content = convert_html_to_markdown(html_str)

    # md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/code/extraction/generated/run_extract_notebook_texts/GenAI_Agents/docs/Academic_Task_Learning_Agent_LangGraph.md"
    # md_content: str = load_file(md_file)

    save_file(md_content, f"{output_dir}/md_content.md")

    markdown_tokens = base_parse_markdown(md_content)
    save_file(markdown_tokens, f"{output_dir}/markdown_tokens.json")

    results = derive_by_header_hierarchy(md_content)
    save_file(results, f"{output_dir}/results.json")
