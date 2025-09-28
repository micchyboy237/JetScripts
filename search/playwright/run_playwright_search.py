from jet.search.playwright import PlaywrightSearch
from jet.file.utils import save_file
import asyncio
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def sync_example(query):
    """Demonstrate synchronous usage of PlaywrightSearch."""
    searcher = PlaywrightSearch()
    try:
        result = searcher._run(
            query=query,
            max_results=5,
            search_depth="basic",
            include_images=False,
            topic="general",
            include_favicon=False
        )
        print("Basic search results:")
        print(f"Query: {result['query']}")
        print(f"Found {len(result['results'])} results:")
        for item in result['results']:
            print(f"- {item['title']} ({item['url']})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_1_result.json")
    except Exception as e:
        print(f"Error in basic search: {e}")

    try:
        result = searcher._run(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_images=True,
            topic="news",
            include_favicon=True
        )
        print("\nAdvanced search results:")
        print(f"Query: {result['query']}")
        print(f"Found {len(result['results'])} results:")
        for item in result['results']:
            print(f"- {item['title']} ({item['url']}, Images: {len(item['images'])}, Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_2_result.json")
    except Exception as e:
        print(f"Error in advanced search: {e}")

async def async_example(query):
    """Demonstrate asynchronous usage of PlaywrightSearch."""
    searcher = PlaywrightSearch()
    try:
        result = await searcher._arun(
            query=query,
            max_results=5,
            search_depth="basic",
            include_domains=["example.com"],
            include_images=False,
            topic="general",
            include_favicon=True
        )
        print("\nAsync basic search results:")
        print(f"Query: {result['query']}")
        print(f"Found {len(result['results'])} results:")
        for item in result['results']:
            print(f"- {item['title']} ({item['url']}, Favicon: {item['favicon']})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_3_result.json")
    except Exception as e:
        print(f"Error in async basic search: {e}")

    try:
        result = await searcher._arun(
            query=query,
            max_results=3,
            search_depth="advanced",
            exclude_domains=["wikipedia.org"],
            include_images=True,
            topic="finance",
            include_favicon=False
        )
        print("\nAsync advanced search results:")
        print(f"Query: {result['query']}")
        print(f"Found {len(result['results'])} results:")
        for item in result['results']:
            print(f"- {item['title']} ({item['url']}, Images: {len(item['images'])})")
        save_file(result, f"{OUTPUT_DIR}/example_4_result.json")
    except Exception as e:
        print(f"Error in async advanced search: {e}")

if __name__ == "__main__":
    query = "recent advancements in AI 2025"
    print("Running synchronous examples...")
    sync_example(query)
    print("\nRunning asynchronous examples...")
    asyncio.run(async_example(query))
