import logging
import os

import requests
import trafilatura

# Centralized adapters & config
from jet.adapters.llama_cpp.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.rerank_utils import rerank as adapter_rerank
from trafilatura.settings import use_config

logger = logging.getLogger(__name__)

# SearXNG and Trafilatura have no adapter equivalents; keep local config
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

TRAFA_CONFIG = use_config()
TRAFA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
TRAFA_CONFIG.set("DEFAULT", "NO_FALLBACK", "False")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def web_search(query: str, num_results: int = 10) -> list[dict]:
    """
    Production SearXNG integration with error handling and structured output.
    Returns results formatted for downstream reranking.
    """
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "pageno": 1,
                "categories": "general",
                "language": "en",
            },
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "engine": r.get("engine", "unknown"),
                }
            )
        logger.info(f"Search returned {len(results)} results for: {query}")
        return results

    except requests.exceptions.Timeout:
        logger.error(f"SearXNG timeout for query: {query}")
        return []
    except Exception as e:
        logger.error(f"SearXNG error: {e}")
        return []


def _truncate_content(content: str, max_chars: int) -> str:
    """Intelligently truncate at paragraph boundaries."""
    if len(content) <= max_chars:
        return content.strip()

    truncated = content[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.7:
        return truncated[:last_para] + "\n\n[... content truncated ...]"
    return truncated + "\n\n[... content truncated ...]"


def read_url(url: str, max_chars: int = 8000) -> str:
    """
    Robust URL extraction with multi-strategy fallback.
    1. Try trafilatura with default config
    2. Fallback to raw requests with browser headers + manual parse
    3. Return structured error if all fail
    """
    # Strategy 1: trafilatura fetch_url
    try:
        downloaded = trafilatura.fetch_url(url, config=TRAFA_CONFIG)
        if downloaded:
            content = trafilatura.extract(
                downloaded,
                config=TRAFA_CONFIG,
                output_format="markdown",
                include_tables=True,
                include_links=True,
                deduplicate=True,
            )
            if content and len(content.strip()) > 100:
                logger.info(f"[Strategy 1] Extracted {len(content)} chars from {url}")
                return _truncate_content(content, max_chars)
    except Exception as e:
        logger.warning(f"[Strategy 1] Failed for {url}: {e}")

    # Strategy 2: raw requests + trafilatura extract
    try:
        logger.info(f"[Strategy 2] Retrying {url} with browser headers")
        resp = requests.get(
            url, headers=BROWSER_HEADERS, timeout=20, allow_redirects=True
        )
        resp.raise_for_status()

        if not resp.encoding or resp.encoding == "ISO-8859-1":
            resp.encoding = resp.apparent_encoding

        content = trafilatura.extract(
            resp.text,
            output_format="markdown",
            include_tables=True,
            include_links=True,
        )
        if content and len(content.strip()) > 100:
            logger.info(f"[Strategy 2] Extracted {len(content)} chars via raw requests")
            return _truncate_content(content, max_chars)

    except Exception as e:
        logger.warning(f"[Strategy 2] Failed for {url}: {e}")

    logger.error(f"All extraction strategies failed for {url}")
    return (
        f"[ERROR] Could not extract readable content from {url}. "
        "Site may require JavaScript rendering or block automated access."
    )


def rerank_results(query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
    """
    Rerank search results using the centralized llama.cpp rerank adapter.
    Maps adapter output back to original document dicts with relevance scores.
    """
    if not documents:
        return []

    # Extract text snippets for the reranker; track index mapping
    valid_docs: list[str] = []
    index_map: list[int] = []
    for i, d in enumerate(documents):
        text = (d.get("snippet") or d.get("content") or "").strip()
        if text:
            valid_docs.append(text)
            index_map.append(i)

    if not valid_docs:
        logger.warning("All documents have empty content, skipping reranking")
        return documents[:top_n]

    try:
        logger.info(
            f"Reranking {len(valid_docs)} docs via adapter (model={RERANK_MODEL})"
        )
        # Adapter returns list[RerankResult] sorted by score desc
        ranked = adapter_rerank(query=query, documents=valid_docs, top_n=top_n)

        output: list[dict] = []
        for item in ranked:
            orig_idx = index_map[item["index"]]
            doc = documents[orig_idx].copy()
            doc["relevance_score"] = item["score"]
            output.append(doc)

        logger.info(f"Reranked {len(valid_docs)} valid docs → top {len(output)}")
        return output

    except Exception as e:
        logger.warning(f"Adapter reranker failed, returning unranked top-{top_n}: {e}")
        return documents[:top_n]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEMO_QUERY = "latest solid-state battery breakthroughs 2026"

    print("=" * 70)
    print("🔍 RESEARCH PIPELINE DEMO")
    print(f"   Query:     {DEMO_QUERY}")
    print(f"   SearXNG:   {SEARXNG_URL}")
    print(f"   Reranker:  {RERANK_BASE_URL} ({RERANK_MODEL})")
    print(f"   LLM:       {LLM_BASE_URL} ({LLM_MODEL})")
    print("=" * 70)

    # ── STEP 1: Web Search ────────────────────────────────────────────
    print("\n📡 STEP 1: Web Search via SearXNG")
    raw_results = web_search(DEMO_QUERY, num_results=10)
    if not raw_results:
        print("   ❌ No search results returned. Check SearXNG connectivity.")
        exit(1)
    for i, r in enumerate(raw_results, 1):
        print(f"   [{i}] {r['title'][:60]}... ({r['engine']})")

    # ── STEP 2: Reranking ─────────────────────────────────────────────
    print(f"\n🏆 STEP 2: Reranking {len(raw_results)} candidates")
    ranked_results = rerank_results(DEMO_QUERY, raw_results, top_n=3)
    for i, r in enumerate(ranked_results, 1):
        score = r.get("relevance_score", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"   [{i}] Score={score_str} | {r['title'][:50]}...")
        print(f"       URL: {r['url']}")

    # ── STEP 3: Content Extraction ────────────────────────────────────
    extracted_content = ""
    if ranked_results:
        best_url = ranked_results[0]["url"]
        print(f"\n📄 STEP 3: Extracting content from top result")
        print(f"   URL: {best_url}")
        extracted_content = read_url(best_url, max_chars=4000)

        print(f"\n{'─' * 70}")
        print("📋 EXTRACTED CONTENT PREVIEW (first 1500 chars):")
        print(f"{'─' * 70}")
        preview = extracted_content[:1500]
        if len(extracted_content) > 1500:
            preview += "\n[... truncated for demo ...]"
        print(preview)
        print(f"{'─' * 70}")
        print(f"✅ Total extracted: {len(extracted_content)} chars")
    else:
        print("\n⚠️  No results after reranking. Try a different query.")
        exit(0)

    # ── STEP 4: LLM Synthesis ─────────────────────────────────────────
    print(f"\n🤖 STEP 4: LLM Synthesis (model={LLM_MODEL})")
    if extracted_content and not extracted_content.startswith("[ERROR]"):
        llm_prompt = (
            f"Based on the following research content, answer this question:\n"
            f"'{DEMO_QUERY}'\n\n"
            f"## Research Content\n{extracted_content}\n\n"
            f"Provide a concise, well-structured summary with key findings."
        )
        logger.info(f"Sending {len(llm_prompt)} char prompt to LLM")
        result = chat(prompt=llm_prompt, model=LLM_MODEL)

        print(f"\n{'═' * 70}")
        print("💡 LLM RESPONSE:")
        print(f"{'═' * 70}")
        print(result.content)
        print(f"{'═' * 70}")
        if result.usage:
            print(
                f"📊 Tokens: {result.usage.get('prompt_tokens', '?')} in / "
                f"{result.usage.get('completion_tokens', '?')} out | "
                f"Finish: {result.finish_reason}"
            )
    else:
        print("   ⚠️  Skipping LLM step: no valid content to synthesize.")

    print("\n✨ Demo complete. Integrate these functions into ResearchAgent.run()")
