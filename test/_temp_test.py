import asyncio

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

QUERY = "chunking strategies for PDF tables"
ROOT_URL = "https://docs.unstructured.io"
MAX_PAGES = 10  # Safety limit


async def crawl_until_satisfied():
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=200,  # Skip thin pages
        exclude_external_links=True,  # INNER LINKS ONLY
        keep_data_attributes=True,
        extraction_strategy=LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",
            api_token="YOUR_API_KEY",
            instruction=f"""Extract content relevant to: '{QUERY}'.
            Return JSON: {{"relevant": bool, "summary": str, "key_points": list}}.
            If page has NO useful info about this query, set relevant=false.""",
            schema={
                "type": "object",
                "properties": {
                    "relevant": {"type": "boolean"},
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
    )

    collected_context = []
    visited = set()
    queue = [ROOT_URL]

    async with AsyncWebCrawler() as crawler:
        while queue and len(collected_context) < 3 and len(visited) < MAX_PAGES:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            result = await crawler.arun(url=url, config=config)
            extracted = result.extracted_content  # Parsed JSON from LLM

            if extracted and extracted.get("relevant"):
                collected_context.append(
                    {
                        "url": url,
                        "markdown": result.markdown[:3000],
                        "summary": extracted["summary"],
                        "key_points": extracted["key_points"],
                    }
                )
                print(f"✅ RELEVANT [{len(collected_context)}]: {url}")
            else:
                print(f"❌ SKIP: {url}")

            # Discover inner links for next iteration
            if result.links and result.links.get("internal"):
                for link in result.links["internal"]:
                    href = link.get("href", "")
                    if href.startswith("/") or ROOT_URL in href:
                        full = (
                            href
                            if href.startswith("http")
                            else ROOT_URL.rstrip("/") + href
                        )
                        if full not in visited:
                            queue.append(full)

    print(f"\n=== FINAL RAG CONTEXT ({len(collected_context)} pages) ===")
    for ctx in collected_context:
        print(f"\n📄 {ctx['url']}")
        print(f"Summary: {ctx['summary']}")
        print(f"Key Points: {ctx['key_points']}")


asyncio.run(crawl_until_satisfied())
