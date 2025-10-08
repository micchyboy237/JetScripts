from jet.search.playwright import PlaywrightCrawl
from jet.file.utils import save_file
import asyncio
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def sync_example(url):
    """Demonstrate synchronous usage of PlaywrightCrawl."""
    # Initialize the tool
    crawler = PlaywrightCrawl()

    # Example 1: Basic crawl of a website
    result = crawler._run(
        url=url,
        max_depth=2,
        max_breadth=10,
        limit=20,
        extract_depth="basic",
        format="markdown"
    )
    print("Basic crawl results:")
    print(f"Base URL: {result['base_url']}")
    print(f"Found {len(result['results'])} pages:")
    for item in result['results']:
        print(f"- {item['url']} (Content length: {len(item['raw_content'])})")
    print(f"Response time: {result['response_time']:.2f} seconds")
    save_file(result, f"{OUTPUT_DIR}/example_1_result.json")

    # Example 2: Crawl with specific path, category, and images
    result = crawler._run(
        url=url,
        max_depth=1,
        # select_paths=["/blog/.*"],
        # categories=["Blogs"],
        include_images=True,
        extract_depth="advanced"
    )
    print("\nBlog-specific crawl results:")
    print(f"Base URL: {result['base_url']}")
    print(f"Found {len(result['results'])} blog pages:")
    for item in result['results']:
        print(f"- {item['url']} (Images: {len(item['images'])})")
    save_file(result, f"{OUTPUT_DIR}/example_2_result.json")

async def async_example(url):
    """Demonstrate asynchronous usage of PlaywrightCrawl."""
    crawler = PlaywrightCrawl()

    # Example 3: Async crawl with domain restrictions and favicon
    result = await crawler._arun(
        url=url,
        max_depth=2,
        # select_domains=["^example\\.com$"],
        # exclude_paths=["/admin/.*"],
        allow_external=False,
        include_favicon=True,
        format="text"
    )
    print("\nAsync crawl results:")
    print(f"Base URL: {result['base_url']}")
    print(f"Found {len(result['results'])} pages:")
    for item in result['results']:
        print(f"- {item['url']} (Favicon: {item['favicon']})")
    print(f"Response time: {result['response_time']:.2f} seconds")
    save_file(result, f"{OUTPUT_DIR}/example_3_result.json")

    # Example 4: Async crawl with instructions and advanced extraction
    result = await crawler._arun(
        url=url,
        instructions="API documentation",
        categories=["Documentation"],
        limit=15,
        extract_depth="advanced",
        include_images=True
    )
    print("\nAPI documentation crawl results:")
    print(f"Base URL: {result['base_url']}")
    print(f"Found {len(result['results'])} documentation pages:")
    for item in result['results']:
        print(f"- {item['url']} (Content length: {len(item['raw_content'])})")
    save_file(result, f"{OUTPUT_DIR}/example_4_result.json")

if __name__ == "__main__":
    url = "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"

    # Run synchronous examples
    print("Running synchronous examples...")
    sync_example(url)

    # Run asynchronous examples
    print("\nRunning asynchronous examples...")
    asyncio.run(async_example(url))
