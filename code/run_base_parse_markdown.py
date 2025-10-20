import os
import shutil

from jet.code.markdown_utils import extract_markdown_links, remove_markdown_links
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")

    body_links, cleaned_text = extract_markdown_links("[ Battlefield 6 ](/db/video-game/battlefield-6/)", ignore_links=True)
    text_no_links = remove_markdown_links("[ Battlefield 6 ](/db/video-game/battlefield-6/)", remove_text=True)


    save_file(body_links, f"{OUTPUT_DIR}/body_links.json")
    save_file(cleaned_text, f"{OUTPUT_DIR}/cleaned_text.txt")
    save_file(text_no_links, f"{OUTPUT_DIR}/text_no_links.txt")
