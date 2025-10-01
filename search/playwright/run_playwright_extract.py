from typing import List, TypedDict
from jet._token.token_utils import token_counter
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.search.playwright import PlaywrightExtract
from jet.file.utils import save_file
import os
import shutil

from jet.search.playwright.playwright_extract import convert_html_to_markdown

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class ContextItem(TypedDict):
    doc_idx: int
    tokens: int
    text: str

def sync_example(urls):
    """Demonstrate synchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = extractor._run(
            urls=urls,
            extract_depth="basic",
            include_images=False,
            include_favicon=False,
            format="markdown"
        )
        print("Basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Content length: {len(item['raw_content'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_1_result.json")
    except Exception as e:
        print(f"Error in basic extract: {e}")

    try:
        result = extractor._run(
            urls=urls,
            extract_depth="advanced",
            include_images=True,
            include_favicon=True,
            format="text"
        )
        print("\nAdvanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Images: {len(item['images'])}, Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_2_result.json")
    except Exception as e:
        print(f"Error in advanced extract: {e}")

async def async_example(urls):
    """Demonstrate asynchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="basic",
            include_images=True,
            include_favicon=False,
            format="markdown"
        )
        print("\nAsync basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Images: {len(item['images'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_3_result.json")
    except Exception as e:
        print(f"Error in async basic extract: {e}")

    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="advanced",
            include_images=False,
            include_favicon=True,
            format="text"
        )
        print("\nAsync advanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_4_result.json")
    except Exception as e:
        print(f"Error in async advanced extract: {e}")

def extract_contexts(html: str, url: str, model: str) -> List[ContextItem]:
    md_content = convert_html_to_markdown(html, ignore_links=False)
    original_docs = derive_by_header_hierarchy(md_content, ignore_links=True)
    contexts: List[ContextItem] = []
    for idx, doc in enumerate(original_docs):
        doc["source"] = url
        text = doc["header"] + "\n\n" + doc["content"]
        contexts.append({
            "doc_idx": idx,
            "tokens": token_counter(text, model),
            "text": text,
        })
    return contexts

def scrape_urls_data(urls: List[str], model: str):
    extractor = PlaywrightExtract()
    result_stream = extractor._stream(
        urls=urls,
        extract_depth="advanced",
        include_images=True,
        include_favicon=True,
        format="text"
    )
    print("\nAdvanced extract results stream:")
    count = 0
    for result in result_stream:
        count += 1
        meta = result.pop("meta")
        print(f"URL: {result['url']} (Images: {len(result['images'])}, Favicon: {result['favicon']})")

        contexts = extract_contexts(meta["html"], result['url'], model)
        save_file(result, f"{OUTPUT_DIR}/stream_1/results.json")
        save_file(contexts, f"{OUTPUT_DIR}/stream_1/contexts.json")
        save_file({
            "tokens": meta["tokens"],
        }, f"{OUTPUT_DIR}/stream_1/info.json")
        save_file(meta["analysis"], f"{OUTPUT_DIR}/stream_1/analysis.json")
        save_file(meta["text_links"], f"{OUTPUT_DIR}/stream_1/text_links.json")
        save_file(meta["image_links"], f"{OUTPUT_DIR}/stream_1/image_links.json")
        save_file(meta["html"], f"{OUTPUT_DIR}/stream_1/page.html")
        save_file(meta["markdown"], f"{OUTPUT_DIR}/stream_1/markdown.md")
        save_file(meta["md_tokens"], f"{OUTPUT_DIR}/stream_1/md_tokens.json")
        save_file(meta["screenshot"], f"{OUTPUT_DIR}/stream_1/screenshot.png")

if __name__ == "__main__":
    urls = [
        "https://docs.tavily.com/documentation/api-reference/endpoint/crawl",
    ]
    model = "qwen3:4b-q4_K_M"

    print("Running stream examples...")
    scrape_urls_data(urls, model)
    # print("Running synchronous examples...")
    # sync_example(urls)
    # print("\nRunning asynchronous examples...")
    # asyncio.run(async_example(urls))
