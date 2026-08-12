import asyncio
import json
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
# NOTE: Reranking uses raw httpx since llama.cpp /rerank is not part of the OpenAI SDK spec

MAX_ITERATIONS = 5
MAX_PAGES_PER_ITER = 3
RELEVANCE_THRESHOLD = 0.35


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


async def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank using local cross-encoder via raw httpx (not OpenAI SDK)."""
    if not chunks:
        return []

    docs = [c["content"] for c in chunks]

    # ✅ FIX: Use httpx directly instead of AsyncOpenAI.post()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{RERANK_BASE_URL}/rerank",
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
            chunks[idx]["score"] = r["relevance_score"]

    return sorted(chunks, key=lambda x: x["score"], reverse=True)


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
            all_new_links.extend([l for l in links if l not in visited])

    ranked_chunks = await rerank_chunks(state["query"], new_chunks)
    relevant = [c for c in ranked_chunks if c["score"] >= RELEVANCE_THRESHOLD]

    existing_urls = {k["url"] for k in state["knowledge_base"]}
    merged_kb = state["knowledge_base"] + [
        c for c in relevant if c["url"] not in existing_urls
    ]

    return {
        "knowledge_base": merged_kb,
        "visited_urls": visited,
        "pending_urls": list(set(all_new_links) - visited)[:10],
        "iteration": state["iteration"] + 1,
    }


async def evaluate_node(state: WebSwarmState) -> dict:
    """LLM evaluates if current KB sufficiently answers the query."""
    kb_summary = "\n---\n".join(
        f"[{c['url']}] (score:{c['score']:.2f})\n{c['content'][:500]}"
        for c in state["knowledge_base"][:10]
    )

    prompt = f"""You are a RAG evaluation agent. Determine if the retrieved context sufficiently answers the query.

QUERY: {state["query"]}
ROOT URL: {state["root_url"]}
ITERATION: {state["iteration"]}/{MAX_ITERATIONS}

RETRIEVED CONTEXT:
{kb_summary}

Respond with ONLY valid JSON:
{{"evaluation": "sufficient|insufficient|irrelevant", "reasoning": "brief explanation"}}

Rules:
- "sufficient": Context directly answers the query with high confidence
- "insufficient": Partial answer exists but needs more depth/breadth AND pending_urls remain
- "irrelevant": Retrieved content is off-topic OR max iterations reached without answer"""

    resp = await LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)
    return {"evaluation": result.get("evaluation", "insufficient")}


async def synthesize_node(state: WebSwarmState) -> dict:
    """Generate final answer from accumulated knowledge."""
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
{context}"""

    resp = await LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return {"final_answer": resp.choices[0].message.content}


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
