# examples/04_llm_usage/01_llm_schema_fallback.py
"""
Pattern: Try fast CSS schema first → if it fails / gives poor results → fallback to LLM
This is one of the most practical hybrid approaches.
"""

import asyncio
import json
import os
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config

SCHEMA_PATH = Path("cache/fallback_article_schema.json")


async def try_css_extraction(crawler, url):
    if not SCHEMA_PATH.exists():
        return None, "No cached schema"

    schema = json.loads(SCHEMA_PATH.read_text())
    strategy = JsonCssExtractionStrategy(schema)

    cfg = CrawlerRunConfig(extraction_strategy=strategy, cache_mode=CacheMode.BYPASS)

    result = await crawler.arun(url, config=cfg)
    if result.success and result.extracted_content:
        try:
            data = json.loads(result.extracted_content)
            if len(data) > 0 and all(k in data[0] for k in ["title", "content"]):
                return data, "CSS schema succeeded"
        except:
            pass
    return None, "CSS schema failed or empty"


async def llm_fallback(crawler, url):
    llm_cfg = get_llm_config(
        provider="openai/qwen3-instruct-2507:4b",  # or groq/llama-3.1-70b, etc.
        base_url=os.getenv("LLAMA_CPP_LLM_MODEL"),
        temperature=0.1,
        max_tokens=1800,
    )

    strategy = LLMExtractionStrategy(
        llm_config=llm_cfg,
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "publish_date": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "reading_time_minutes": {"type": "integer"},
            },
            "required": ["title", "content"],
        },
        extraction_type="schema",
        instruction=(
            "Extract the main article content cleanly. "
            "Remove ads, navigation, related links, comments. "
            "Keep only title, authors, date, body text, tags."
        ),
        apply_chunking=True,
        chunk_token_threshold=1400,
    )

    cfg = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=30,
    )

    result = await crawler.arun(url, config=cfg)
    if result.success and result.extracted_content:
        return json.loads(result.extracted_content), "LLM fallback succeeded"
    return None, result.error_message or "LLM failed"


async def main():
    url = "https://example.com/long-read-article"  # ← replace

    async with AsyncWebCrawler(verbose=True) as crawler:
        # Step 1: fast path
        data, msg = await try_css_extraction(crawler, url)
        if data:
            print(f"✓ CSS succeeded: {msg}")
        else:
            print(f"CSS attempt: {msg} → falling back to LLM")
            data, msg = await llm_fallback(crawler, url)
            print(f"→ {msg}")

        if data:
            print("\nExtracted article:")
            print(f"Title: {data.get('title', '—')}")
            print(f"Authors: {', '.join(data.get('authors', [])) or '—'}")
            print(f"Date: {data.get('publish_date', '—')}")
            print(f"Tags: {', '.join(data.get('tags', [])) or '—'}")
            print(f"Reading time: {data.get('reading_time_minutes', '?')} min\n")
            print("Content preview (first 300 chars):")
            print(data.get("content", "—")[:300] + "…")
        else:
            print("Both methods failed.")


if __name__ == "__main__":
    asyncio.run(main())
