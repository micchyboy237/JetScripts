import os
import shutil
import base64
from typing import Literal
from pathlib import Path
from jet.file.utils import save_file
from jet.scrapers.utils import extract_by_heading_hierarchy
from jet.search.deep_search import rag_search, RagSearchResult
from jet.utils.text import format_sub_source_dir

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def save_image(
    image_data: str | bytes,
    output_path: str | Path,
    image_format: Literal["png", "jpeg", "jpg"] = "png"
) -> None:
    """
    Save raw image bytes or base64-encoded image to file.

    Args:
        image_data: Base64 string (optionally prefixed) or raw image bytes.
        output_path: Where to save the image.
        image_format: Optional, not currently used but could be used for validation.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists

        if isinstance(image_data, bytes):
            # Already raw image bytes → save directly
            with output_path.open("wb") as f:
                f.write(image_data)
        elif isinstance(image_data, str):
            # Base64-encoded string (may include data URI prefix)
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            with output_path.open("wb") as f:
                f.write(image_bytes)
        else:
            raise ValueError("Unsupported image_data type")

    except (base64.binascii.Error, ValueError) as e:
        raise ValueError("Invalid base64 string or image bytes") from e
    except OSError as e:
        raise OSError(f"Failed to save image to {output_path}") from e



def example_usage():
    query = "Latest advancements in AI research 2025"

    print("\nRunning rag_search...")
    rag_search_result: RagSearchResult = rag_search(
        query=query,
        embed_model="all-MiniLM-L12-v2",
        top_k=None,
        threshold=0.0,
        chunk_size=200,
        chunk_overlap=50,
        merge_chunks=False,
        # urls=sample_urls,
        use_cache=False,
        urls_limit=10
    )

    # Example of processing high-score URLs
    for url in rag_search_result['all_urls_with_high_scores'][:2]:  # Show top 2 high-score URLs
        print(f"\nHigh-score URL: {url}")

    url_results = rag_search_result.pop('url_results')
    urls = [r['url'] for r in url_results]
    htmls = [r['html'] for r in url_results]
    screenshots = [r['screenshot'] for r in url_results]

    for k, v in rag_search_result.items():
        save_file(v, f"{OUTPUT_DIR}/{k}.json")

    for url, html, screenshot in zip(urls, htmls, screenshots):
        sub_dir_url = format_sub_source_dir(url)
        sub_dir = f"{OUTPUT_DIR}/url_results/{sub_dir_url}"

        header_docs = extract_by_heading_hierarchy(html)
        md_content = "\n\n".join([node.text for node in header_docs])

        save_file(html, f"{sub_dir}/page.html")
        save_file(header_docs, f"{sub_dir}/docs.json")
        save_file(md_content, f"{sub_dir}/markdown.md")
        save_image(screenshot, f"{sub_dir}/screenshot.png")

if __name__ == "__main__":
    example_usage()
