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
# ✅ FIX 1: Lowered threshold - BGE-Reranker via llama.cpp often returns
# scores in 0.0-0.3 range for partially relevant docs. 0.35 was filtering everything.
RELEVANCE_THRESHOLD = 0.10
# ✅ FIX 2: Always keep top-N chunks even if below threshold, so the graph
# never has a completely empty KB on first iteration
MIN_CHUNKS_TO_KEEP = 3

# Set to True for bge-reranker-v2-m3, ms-marco, etc.
# Set to False for jina-reranker, cohere-rerank
APPLY_SIGMOID_NORMALIZATION = True


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
        return []

    docs = [c["content"] for c in chunks]

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
    for r in results:
        idx = r["index"]
        if idx < len(chunks):
            raw_score = r["relevance_score"]
            chunks[idx]["score"] = (
                _sigmoid(raw_score) if APPLY_SIGMOID_NORMALIZATION else raw_score
            )
            chunks[idx]["raw_score"] = raw_score  # Always preserve for debugging

    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # Diagnostic: print score range so you can verify normalization
    if ranked:
        scores = [c["score"] for c in ranked]
        print(
            f"[RERANK] scores={'normalized' if APPLY_SIGMOID_NORMALIZATION else 'raw'} "
            f"min={scores[-1]:.4f} max={scores[0]:.4f} median={scores[len(scores) // 2]:.4f}"
        )

    return ranked


async def fetch_and_parse(url: str) -> tuple[str, list[str]]:
    """Fetch page and extract text + internal links."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "WebSwarmBot/1.0"})
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)[:8000]

        base_domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            full_url = urljoin(url, a["href"])
            if urlparse(full_url).netloc == base_domain and full_url not in links:
                links.append(full_url)

        return text, links[:20]
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return "", []


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

    full_content = ""
    print(f"\n[{label}] ", end="", flush=True)

    stream = await LLM_CLIENT.chat.completions.create(**create_kwargs)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_content += delta
            print(delta, end="", flush=True)

    print("", flush=True)
    return full_content


# --- Graph Nodes ---
async def retrieve_node(state: WebSwarmState) -> dict:
    """Fetch pending URLs, chunk, embed, and rerank."""
    pending = state["pending_urls"][:MAX_PAGES_PER_ITER]
    visited = state["visited_urls"] | set(pending)

    new_chunks = []
    all_new_links = []

    tasks = [fetch_and_parse(url) for url in pending]
    results = await asyncio.gather(*tasks)

    for url, (text, links) in zip(pending, results):
        if text:
            new_chunks.append({"url": url, "content": text, "score": 0.0})
            # ✅ FIX 3: Always propagate discovered links regardless of
            # whether this page's content passed the reranker threshold.
            # Previously, links were only added if text was non-empty,
            # but we also need links from pages that fetched successfully
            # even if their content scored low.
            all_new_links.extend([l for l in links if l not in visited])

    ranked_chunks = await rerank_chunks(state["query"], new_chunks)

    # ✅ FIX 4: Keep at least MIN_CHUNKS_TO_KEEP even if below threshold,
    # OR keep all chunks above threshold, whichever yields more results.
    above_threshold = [c for c in ranked_chunks if c["score"] >= RELEVANCE_THRESHOLD]
    if len(above_threshold) < MIN_CHUNKS_TO_KEEP and ranked_chunks:
        relevant = ranked_chunks[: max(MIN_CHUNKS_TO_KEEP, len(above_threshold))]
        print(
            f"[RETRIEVE] Only {len(above_threshold)} chunks above threshold "
            f"({RELEVANCE_THRESHOLD}), keeping top {len(relevant)} by score"
        )
    else:
        relevant = above_threshold

    existing_urls = {k["url"] for k in state["knowledge_base"]}
    merged_kb = state["knowledge_base"] + [
        c for c in relevant if c["url"] not in existing_urls
    ]

    print(
        f"[RETRIEVE] Iteration {state['iteration'] + 1}: "
        f"fetched={len(new_chunks)}, relevant={len(relevant)}, "
        f"kb_total={len(merged_kb)}, pending={len(all_new_links)}"
    )

    return {
        "knowledge_base": merged_kb,
        "visited_urls": visited,
        "pending_urls": list(set(all_new_links) - visited)[:10],
        "iteration": state["iteration"] + 1,
    }


async def evaluate_node(state: WebSwarmState) -> dict:
    """LLM evaluates if current KB sufficiently answers the query (streamed)."""
    kb_summary = "\n---\n".join(
        f"[{c['url']}] (score:{c['score']:.2f})\n{c['content'][:500]}"
        for c in state["knowledge_base"][:10]
    )

    prompt = f"""You are a RAG evaluation agent. Determine if the retrieved context sufficiently answers the query.

QUERY: {state["query"]}
ROOT URL: {state["root_url"]}
ITERATION: {state["iteration"]}/{MAX_ITERATIONS}

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
    return {"evaluation": result.get("evaluation", "insufficient")}


async def synthesize_node(state: WebSwarmState) -> dict:
    """Generate final answer from accumulated knowledge (streamed)."""
    context = "\n\n===SOURCE===".join(
        f"URL: {c['url']}\nRelevance: {c['score']:.2f}\n{c['content']}"
        for c in sorted(
            state["knowledge_base"], key=lambda x: x["score"], reverse=True
        )[:8]
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
    if state["evaluation"] == "sufficient":
        return "synthesize"
    if state["evaluation"] == "irrelevant" or state["iteration"] >= MAX_ITERATIONS:
        return "synthesize"
    if not state["pending_urls"]:
        return "synthesize"
    return "retrieve"


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
    print(f"\n{'=' * 60}")
    print(f"ANSWER ({result['iterations']} iters, {result['pages_visited']} pages):")
    print(result["answer"])
    print(f"\nSOURCES: {result['sources']}")
