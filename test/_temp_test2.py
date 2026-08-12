import asyncio
import os

from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

QUERY = "chunking strategies for PDF tables"
ROOT_URL = "https://docs.unstructured.io"
MAX_PAGES = 10
TARGET_RELEVANT = 3


class RAGExtraction(BaseModel):
    """Schema for LLM-extracted RAG context."""

    relevant: bool = Field(
        description="Whether the page contains useful info about the query"
    )
    summary: str = Field(description="Concise summary of relevant content")
    key_points: list[str] = Field(description="List of key takeaways")


async def main() -> None:
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))
    collected_count = 0

    crawler = PlaywrightCrawler(
        max_requests_per_crawl=MAX_PAGES,
        headless=True,
        browser_type="chromium",
    )

    @crawler.router.default_handler
    async def handle_page(context: PlaywrightCrawlingContext) -> None:
        nonlocal collected_count

        # Stop early if we have enough relevant pages
        if collected_count >= TARGET_RELEVANT:
            context.log.info(
                f"Target reached ({TARGET_RELEVANT}), skipping {context.request.url}"
            )
            return

        # Extract page text for LLM (limit to ~3000 chars to control tokens)
        text = await context.page.inner_text("body")
        truncated = text[:3000]

        # Skip thin pages (equivalent to word_count_threshold=200)
        if len(truncated.split()) < 200:
            context.log.info(f"❌ SKIP (thin): {context.request.url}")
            await context.enqueue_links(strategy="same-domain")
            return

        # Call LLM with structured output
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured data for RAG pipelines.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extract content relevant to: '{QUERY}'.\n"
                            f"If the page has NO useful info, set relevant=false.\n\n"
                            f"Page content:\n{truncated}"
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rag_extraction",
                        "schema": RAGExtraction.model_json_schema(),
                    },
                },
            )
            extracted = RAGExtraction.model_validate_json(
                response.choices[0].message.content or "{}"
            )
        except Exception as e:
            context.log.error(f"LLM error on {context.request.url}: {e}")
            await context.enqueue_links(strategy="same-domain")
            return

        if extracted.relevant:
            collected_count += 1
            await context.push_data(
                {
                    "url": context.request.url,
                    "markdown": truncated,
                    "summary": extracted.summary,
                    "key_points": extracted.key_points,
                }
            )
            context.log.info(f"✅ RELEVANT [{collected_count}]: {context.request.url}")
        else:
            context.log.info(f"❌ SKIP (irrelevant): {context.request.url}")

        # Only continue crawling if we haven't reached target
        if collected_count < TARGET_RELEVANT:
            await context.enqueue_links(strategy="same-domain")

    await crawler.run([ROOT_URL])

    # Print final results from dataset
    dataset = await crawler.get_dataset()
    items = await dataset.get_data()
    print(f"\n=== FINAL RAG CONTEXT ({len(items.items)} pages) ===")
    for item in items.items:
        print(f"\n📄 {item['url']}")
        print(f"Summary: {item['summary']}")
        print(f"Key Points: {item['key_points']}")


if __name__ == "__main__":
    asyncio.run(main())
