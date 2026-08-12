import asyncio
import json
import math
from typing import Annotated, Literal, Sequence, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL_LG,
    EMBED_MODEL_LG,
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import AsyncOpenAI

# --- Configuration for Local llama.cpp Servers ---
LLM_CLIENT = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="local")
EMBED_CLIENT = AsyncOpenAI(base_url=EMBED_BASE_URL_LG, api_key="local")

MAX_ITERATIONS = 5
MAX_PAGES_PER_ITER = 3
RELEVANCE_THRESHOLD = 0.10
MIN_CHUNKS_TO_KEEP = 3
APPLY_SIGMOID_NORMALIZATION = True

# ✅ NEW: Allow following redirects to different domains
# Set to True if root URLs may redirect to external docs sites
FOLLOW_CROSS_DOMAIN_LINKS = True


# --- State Definition ---
class WebSwarmState(TypedDict):
    query: str
    root_url: str
    messages: Annotated[Sequence, add_messages]
    visited_urls: set[str]
    pending_urls: list[str]
    knowledge_base: list[dict]
    iteration: int
    evaluation: Literal["sufficient", "insufficient", "irrelevant"]
    final_answer: str | None


# --- Helper Functions for Local Models ---
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from local llama.cpp server."""
    resp = await EMBED_CLIENT.embeddings.create(model=EMBED_MODEL_LG, input=texts)
    return [d.embedding for d in resp.data]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


async def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank using local cross-encoder via raw httpx with optional sigmoid normalization."""
    if not chunks:
        print("[RERANK] ⚠️  No chunks to rerank")
        return []

    docs = [c["content"] for c in chunks]
    print(
        f"[RERANK] Sending {len(docs)} docs to {RERANK_BASE_URL}/rerank (model={RERANK_MODEL})"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{RERANK_BASE_URL.rstrip('/')}/rerank",
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": docs,
                "top_n": len(docs),
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    print(f"[RERANK] Received {len(results)} results from server")

    for r in results:
        idx = r["index"]
        if idx < len(chunks):
            raw_score = r["relevance_score"]
            normalized = (
                _sigmoid(raw_score) if APPLY_SIGMOID_NORMALIZATION else raw_score
            )
            chunks[idx]["score"] = normalized
            chunks[idx]["raw_score"] = raw_score

    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # ✅ DEBUG: Per-chunk score breakdown
    if ranked:
        print(
            f"[RERANK] Score distribution ({'normalized' if APPLY_SIGMOID_NORMALIZATION else 'raw'}):"
        )
        for i, c in enumerate(ranked):
            content_preview = c["content"][:80].replace("\n", " ")
            print(
                f"  [{i}] score={c['score']:.4f} raw={c.get('raw_score', 'N/A')} "
                f'url={c["url"]} preview="{content_preview}..."'
            )
    else:
        print("[RERANK] ⚠️  No ranked results returned")

    return ranked


async def fetch_and_parse(url: str) -> tuple[str, list[str], dict]:
    """Fetch page and extract text + internal links. Returns (text, links, metadata)."""
    meta = {
        "url": url,
        "status": None,
        "final_url": None,
        "text_len": 0,
        "links_found": 0,
        "links_filtered": 0,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "WebSwarmBot/1.0"})
            resp.raise_for_status()
            meta["status"] = resp.status_code
            meta["final_url"] = str(resp.url)

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)[:8000]
        meta["text_len"] = len(text)

        # ✅ DEBUG: Log redirect detection
        if meta["final_url"] != url:
            print(f"[FETCH] ↪ Redirect detected: {url} → {meta['final_url']}")

        base_domain = urlparse(url).netloc
        final_domain = urlparse(meta["final_url"]).netloc

        all_links = []
        for a in soup.find_all("a", href=True):
            full_url = urljoin(str(resp.url), a["href"])  # Use final URL for resolving
            parsed = urlparse(full_url)
            if parsed.scheme in ("http", "https") and full_url not in all_links:
                all_links.append(full_url)

        meta["links_found"] = len(all_links)

        # Filter links based on domain policy
        if FOLLOW_CROSS_DOMAIN_LINKS:
            # Allow cross-domain but still filter out obvious non-content URLs
            filtered = [
                l
                for l in all_links
                if not any(
                    skip in l.lower()
                    for skip in ["#", "javascript:", "mailto:", ".pdf", ".zip"]
                )
            ]
        else:
            filtered = [l for l in all_links if urlparse(l).netloc == base_domain]

        meta["links_filtered"] = len(filtered)
        links = filtered[:20]

        print(
            f"[FETCH] ✅ {url} | status={meta['status']} | "
            f"text={meta['text_len']} chars | links={meta['links_found']} found, "
            f"{meta['links_filtered']} after filter | final_domain={final_domain}"
        )

        return text, links, meta
    except Exception as e:
        meta["error"] = str(e)
        print(f"[FETCH] ❌ {url} | error={e}")
        return "", [], meta


async def stream_llm_completion(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.95,
    presence_penalty: float = 1.5,
    response_format: dict | None = None,
    label: str = "LLM",
) -> str:
    """Stream LLM completion with flushed output and enable_thinking disabled."""
    create_kwargs: dict = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if response_format is not None:
        create_kwargs["response_format"] = response_format

    prompt_chars = len(prompt)
    print(
        f"\n[{label}] Starting generation (prompt={prompt_chars} chars, max_tokens={max_tokens}, temp={temperature})",
        flush=True,
    )

    full_content = ""
    token_count = 0
    print(f"[{label}] ", end="", flush=True)

    stream = await LLM_CLIENT.chat.completions.create(**create_kwargs)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_content += delta
            token_count += 1
            print(delta, end="", flush=True)

    print("", flush=True)
    print(
        f"[{label}] Completed: {token_count} tokens, {len(full_content)} chars",
        flush=True,
    )
    return full_content


# --- Graph Nodes ---
async def retrieve_node(state: WebSwarmState) -> dict:
    """Fetch pending URLs, chunk, embed, and rerank."""
    pending = state["pending_urls"][:MAX_PAGES_PER_ITER]
    visited = state["visited_urls"] | set(pending)

    print(f"\n{'─' * 60}")
    print(f"[RETRIEVE] ▶ Iteration {state['iteration'] + 1}/{MAX_ITERATIONS}")
    print(f"[RETRIEVE] Pending URLs: {pending}")
    print(f"[RETRIEVE] Already visited: {len(state['visited_urls'])} URLs")

    new_chunks = []
    all_new_links = []
    fetch_metadata = []

    tasks = [fetch_and_parse(url) for url in pending]
    results = await asyncio.gather(*tasks)

    for url, (text, links, meta) in zip(pending, results):
        fetch_metadata.append(meta)
        if text:
            new_chunks.append(
                {"url": meta.get("final_url", url), "content": text, "score": 0.0}
            )
            all_new_links.extend([l for l in links if l not in visited])
        elif meta.get("error"):
            print(f"[RETRIEVE] ⚠️  Skipping {url}: fetch failed")
        else:
            print(
                f"[RETRIEVE] ⚠️  Skipping {url}: empty content ({meta['text_len']} chars)"
            )

    print(
        f"[RETRIEVE] Fetched {len(new_chunks)} pages with content, discovered {len(all_new_links)} new links"
    )

    ranked_chunks = await rerank_chunks(state["query"], new_chunks)

    above_threshold = [c for c in ranked_chunks if c["score"] >= RELEVANCE_THRESHOLD]
    if len(above_threshold) < MIN_CHUNKS_TO_KEEP and ranked_chunks:
        relevant = ranked_chunks[: max(MIN_CHUNKS_TO_KEEP, len(above_threshold))]
        print(
            f"[RETRIEVE] ⚡ Only {len(above_threshold)} chunks above threshold "
            f"({RELEVANCE_THRESHOLD}), keeping top {len(relevant)} by score"
        )
    else:
        relevant = above_threshold
        print(
            f"[RETRIEVE] ✓ {len(relevant)} chunks above threshold ({RELEVANCE_THRESHOLD})"
        )

    existing_urls = {k["url"] for k in state["knowledge_base"]}
    new_relevant = [c for c in relevant if c["url"] not in existing_urls]
    merged_kb = state["knowledge_base"] + new_relevant

    deduped_pending = list(set(all_new_links) - visited)[:10]

    print(
        f"[RETRIEVE] Summary: fetched={len(new_chunks)}, relevant={len(relevant)}, "
        f"new_to_kb={len(new_relevant)}, kb_total={len(merged_kb)}, "
        f"next_pending={len(deduped_pending)}"
    )
    if not deduped_pending and state["iteration"] == 0:
        print(
            "[RETRIEVE] ⚠️  WARNING: No pending URLs after first iteration! "
            "Check FOLLOW_CROSS_DOMAIN_LINKS or root URL validity."
        )

    return {
        "knowledge_base": merged_kb,
        "visited_urls": visited,
        "pending_urls": deduped_pending,
        "iteration": state["iteration"] + 1,
    }


async def evaluate_node(state: WebSwarmState) -> dict:
    """LLM evaluates if current KB sufficiently answers the query (streamed)."""
    print(
        f"\n[EVALUATE] Assessing KB size={len(state['knowledge_base'])} at iteration {state['iteration']}"
    )

    kb_summary = "\n---\n".join(
        f"[{c['url']}] (score:{c['score']:.4f}, raw:{c.get('raw_score', 'N/A')})\n{c['content'][:500]}"
        for c in state["knowledge_base"][:10]
    )

    has_pending = bool(state.get("pending_urls"))
    print(
        f"[EVALUATE] Pending URLs available: {has_pending} ({len(state.get('pending_urls', []))} remaining)"
    )

    prompt = f"""You are a RAG evaluation agent. Determine if the retrieved context sufficiently answers the query.

QUERY: {state["query"]}
ROOT URL: {state["root_url"]}
ITERATION: {state["iteration"]}/{MAX_ITERATIONS}
PENDING_URLS_AVAILABLE: {has_pending}

RETRIEVED CONTEXT:
{kb_summary if kb_summary else "(No relevant context retrieved yet)"}

Respond with ONLY valid JSON:
{{"evaluation": "sufficient|insufficient|irrelevant", "reasoning": "brief explanation"}}

Rules:
- "sufficient": Context directly answers the query with high confidence
- "insufficient": Partial answer exists but needs more depth/breadth AND pending_urls remain
- "irrelevant": Retrieved content is off-topic OR max iterations reached without answer"""

    content = await stream_llm_completion(
        prompt=prompt,
        temperature=0,
        response_format={"type": "json_object"},
        label="EVALUATE",
    )

    result = json.loads(content)
    evaluation = result.get("evaluation", "insufficient")
    reasoning = result.get("reasoning", "unknown")
    print(f"[EVALUATE] Decision: {evaluation} | Reason: {reasoning}")

    return {"evaluation": evaluation}


async def synthesize_node(state: WebSwarmState) -> dict:
    """Generate final answer from accumulated knowledge (streamed)."""
    sorted_kb = sorted(state["knowledge_base"], key=lambda x: x["score"], reverse=True)[
        :8
    ]
    print(f"\n[SYNTHESIZE] Generating answer from {len(sorted_kb)} KB entries")
    for i, c in enumerate(sorted_kb):
        print(f"  Source [{i}]: {c['url']} (score={c['score']:.4f})")

    context = "\n\n===SOURCE===".join(
        f"URL: {c['url']}\nRelevance: {c['score']:.4f}\n{c['content']}"
        for c in sorted_kb
    )

    prompt = f"""Using ONLY the provided context, answer the query comprehensively.
Cite sources as [URL]. If context is insufficient, say so explicitly.

QUERY: {state["query"]}

CONTEXT:
{context if context else "(No relevant context was retrieved during the search.)"}"""

    content = await stream_llm_completion(
        prompt=prompt,
        temperature=0.1,
        label="SYNTHESIZE",
    )

    return {"final_answer": content}


# --- Conditional Routing ---
def should_continue(state: WebSwarmState) -> Literal["retrieve", "synthesize", END]:
    """Route based on evaluation and iteration limits."""
    decision = None
    reason = ""

    if state["evaluation"] == "sufficient":
        decision = "synthesize"
        reason = "evaluation=sufficient"
    elif state["evaluation"] == "irrelevant":
        decision = "synthesize"
        reason = "evaluation=irrelevant"
    elif state["iteration"] >= MAX_ITERATIONS:
        decision = "synthesize"
        reason = f"max_iterations={MAX_ITERATIONS} reached"
    elif not state["pending_urls"]:
        decision = "synthesize"
        reason = "no_pending_urls"
    else:
        decision = "retrieve"
        reason = (
            f"evaluation={state['evaluation']}, pending={len(state['pending_urls'])}"
        )

    print(f"\n[ROUTE] → {decision.upper()} ({reason})")
    print(f"{'─' * 60}")
    return decision


# --- Build Graph ---
def build_webswarm_graph():
    workflow = StateGraph(WebSwarmState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue)
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# --- Execution ---
async def run_webswarm(query: str, root_url: str):
    app = build_webswarm_graph()

    print(f"\n{'═' * 60}")
    print(f'[INIT] Query: "{query}"')
    print(f"[INIT] Root URL: {root_url}")
    print(
        f"[INIT] Config: max_iter={MAX_ITERATIONS}, pages/iter={MAX_PAGES_PER_ITER}, "
        f"threshold={RELEVANCE_THRESHOLD}, min_keep={MIN_CHUNKS_TO_KEEP}, "
        f"sigmoid={APPLY_SIGMOID_NORMALIZATION}, cross_domain={FOLLOW_CROSS_DOMAIN_LINKS}"
    )
    print(f"[INIT] LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"[INIT] Reranker: {RERANK_MODEL} @ {RERANK_BASE_URL}")
    print(f"{'═' * 60}")

    initial_state = {
        "query": query,
        "root_url": root_url,
        "messages": [],
        "visited_urls": set(),
        "pending_urls": [root_url],
        "knowledge_base": [],
        "iteration": 0,
        "evaluation": "insufficient",
        "final_answer": None,
    }

    config = {"recursion_limit": MAX_ITERATIONS * 3 + 5}
    result = await app.ainvoke(initial_state, config=config)

    print(f"\n{'═' * 60}")
    print(f"[DONE] Iterations: {result['iteration']}")
    print(f"[DONE] Pages visited: {len(result['visited_urls'])}")
    print(f"[DONE] KB entries: {len(result['knowledge_base'])}")
    print(f"[DONE] Sources: {[k['url'] for k in result['knowledge_base']]}")
    print(f"{'═' * 60}")

    return {
        "answer": result["final_answer"],
        "sources": [k["url"] for k in result["knowledge_base"]],
        "iterations": result["iteration"],
        "pages_visited": len(result["visited_urls"]),
    }


if __name__ == "__main__":
    result = asyncio.run(
        run_webswarm(
            query="What are the deployment options for LangGraph Platform?",
            root_url="https://langchain-ai.github.io/langgraph/",
        )
    )
    print(f"\nANSWER ({result['iterations']} iters, {result['pages_visited']} pages):")
    print(result["answer"])
    print(f"\nSOURCES: {result['sources']}")
