from jet.search.playwright import PlaywrightExtract
from jet.file.utils import save_file
import asyncio
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

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

def stream_example(urls):
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
        save_file(result, f"{OUTPUT_DIR}/stream_1/results.json")
        save_file(meta["analysis"], f"{OUTPUT_DIR}/stream_1/analysis.json")
        save_file(meta["text_links"], f"{OUTPUT_DIR}/stream_1/text_links.json")
        save_file(meta["image_links"], f"{OUTPUT_DIR}/stream_1/image_links.json")
        save_file(meta["markdown"], f"{OUTPUT_DIR}/stream_1/markdown.md")
        save_file(meta["md_tokens"], f"{OUTPUT_DIR}/stream_1/md_tokens.json")
        save_file(meta["screenshot"], f"{OUTPUT_DIR}/stream_1/screenshot.png")

if __name__ == "__main__":
    urls = [
        "https://docs.tavily.com/documentation/api-reference/endpoint/crawl",
    ]
    print("Running stream examples...")
    stream_example(urls)
    print("Running synchronous examples...")
    sync_example(urls)
    print("\nRunning asynchronous examples...")
    asyncio.run(async_example(urls))
