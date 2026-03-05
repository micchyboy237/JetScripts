import shutil
from pathlib import Path

from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html
from jet.code.markdown_utils import convert_html_to_markdown
from jet.code.markdown_utils._preprocessors import extract_markdown_links
from jet.file.utils import load_file, save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    url = "https://missav.ws/en"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/missav_ws_dm223_en/sync_results/page.html"
    html = load_file(html_file)
    html = convert_dl_blocks_to_md(html)
    html = preprocess_html(html)

    md_content = convert_html_to_markdown(html, ignore_links=False)

    links, _ = extract_markdown_links(md_content, base_url=url, ignore_links=False)

    save_file(html, OUTPUT_DIR / "page.html")
    save_file(md_content, OUTPUT_DIR / "md_content.md")
    save_file(links, OUTPUT_DIR / "links.json")
