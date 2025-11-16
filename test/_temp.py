import json
from pathlib import Path
import shutil
from typing import Callable, List, Optional, Tuple
from datetime import datetime
from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.file.utils import save_file
from jet.transformers.object import make_serializable
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from jet.visualization.terminal import display_iterm2_image
import os
from jet.logger import logger, CustomLogger
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, ToolMessage
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool
from typing_extensions import Literal
from langchain_core.documents import Document

# Reuse from 3_compress_context.py
from jet.search.searxng import search_searxng, SearchResult

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
os.makedirs(str(BASE_OUTPUT_DIR), exist_ok=True)
log_file = os.path.join(str(BASE_OUTPUT_DIR), "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")

# Default SearXNG URL (can be overridden via env or config)
DEFAULT_QUERY_URL = "http://jethros-macbook-air.local:3000"


# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
def retrieve_relevant_urls(
    query: str,
    count: int = 5,
    min_score: float = 0.1,
    min_date: Optional[datetime] = None,
    include_sites: Optional[List[str]] = None,
    exclude_sites: Optional[List[str]] = None,
) -> List[str]:
    raw_results: List[SearchResult] = search_searxng(
        query_url=DEFAULT_QUERY_URL,
        query=query,
        min_score=min_score,
        min_date=min_date,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
        use_cache=True,
    )
    save_file(raw_results, f"{str(BASE_OUTPUT_DIR)}/web_all_searxng_results.json")

    filtered: List[SearchResult] = []
    for res in raw_results:
        if not (res.get("url") and res.get("title") and res.get("content")):
            continue
        if len(res["content"]) < 50:
            continue
        filtered.append(res)
    save_file(filtered, f"{str(BASE_OUTPUT_DIR)}/web_filtered_searxng_results.json")

    now = datetime.now()
    def composite_rank(res: SearchResult) -> float:
        base = res.get("score", 0.0)
        recency = 0.0
        pub_str = res.get("publishedDate", "")
        if pub_str:
            try:
                pub_date = datetime.fromisoformat(pub_str.split("T")[0])
                days_old = (now - pub_date).days
                recency = max(0.0, 1.0 - days_old / 365.0) * 0.3
            except Exception:
                pass
        length_boost = min(len(res["content"]) / 1000.0, 1.0) * 0.1
        return base + recency + length_boost

    filtered.sort(key=composite_rank, reverse=True)
    save_file(filtered, f"{str(BASE_OUTPUT_DIR)}/web_sorted_searxng_results.json")
    return [res["url"] for res in filtered[:count]]


def load_documents_from_urls(urls: List[str]) -> List[Document]:
    """Wrapper around WebBaseLoader that returns a flat list of Document objects."""
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# ----------------------------------------------------------------------
# NEW: rag_retrieval_with_scores (returns (Document, float) tuples)
# ----------------------------------------------------------------------
def rag_retrieval_with_scores(
    query: str,
    search_count: int = 5,
    min_score: float = 0.1,
    min_date: Optional[datetime] = None,
    include_sites: Optional[List[str]] = None,
    exclude_sites: Optional[List[str]] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    embedding_model: str = "embeddinggemma",
    cache_dir: Optional[str] = None,
    top_k: int = 5,
) -> Tuple[Callable[[str], List[Tuple[Document, float]]], InMemoryVectorStore]:
    """
    Build a RAG retriever that returns (Document, similarity_score) tuples.

    Args:
        query: Initial search query to fetch relevant URLs.
        search_count: Number of URLs to retrieve.
        min_score: Minimum SearXNG relevance score.
        min_date/include_sites/exclude_sites: SearXNG filters.
        chunk_size/chunk_overlap: Document splitting.
        embedding_model: Llama.cpp embedding model.
        cache_dir: Optional directory to cache URLs/chunks.
        top_k: Number of (doc, score) pairs to return per query.

    Returns:
        Tuple[Callable[[str], List[Tuple[Document, float]]], InMemoryVectorStore]
    """
    # 1. Retrieve URLs
    urls = retrieve_relevant_urls(
        query=query,
        count=search_count,
        min_score=min_score,
        min_date=min_date,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
    )

    if not urls:
        logger.warning(f"[rag_retrieval_with_scores] No URLs found for query: '{query}'")
        def empty_retriever(q: str) -> List[Tuple[Document, float]]:
            return []
        # Returning empty retriever and dummy vectorstore
        return empty_retriever, None

    # 2. Load documents
    docs_list = load_documents_from_urls(urls)
    save_file(docs_list, f"{str(BASE_OUTPUT_DIR)}/docs.json")

    # 3. Chunk
    chunked_docs = chunk_documents(docs_list, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_file(chunked_docs, f"{str(BASE_OUTPUT_DIR)}/chunks.json")

    # 4. Embed + Vector Store
    embeddings = EmbedLlamaCpp(model=embedding_model)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
    )

    # 5. Optional caching
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        save_file(urls, f"{cache_dir}/retrieved_urls.json")
        save_file(
            [{"content": d.page_content, "metadata": d.metadata} for d in chunked_docs],
            f"{cache_dir}/chunked_docs.json",
        )

    # 6. Retriever with scores
    def retrieve_with_scores(user_query: str) -> List[Tuple[Document, float]]:
        """Return top-k documents with similarity scores (lower = more similar)."""
        return vectorstore.similarity_search_with_score(
            query=user_query,
            k=top_k,
        )

    return retrieve_with_scores, vectorstore


# ----------------------------------------------------------------------
# Updated Example: uses rag_retrieval_with_scores
# ----------------------------------------------------------------------
def example_4_rag_retrieval(top_k: int = 5):
    """Example 4: Dynamic web RAG using rag_retrieval_with_scores()."""
    example_dir = os.path.join(BASE_OUTPUT_DIR, "example_4_rag_retrieval")
    os.makedirs(example_dir, exist_ok=True)
    log_file = f"{example_dir}/main.log"
    logger_local = CustomLogger("example_4_rag_retrieval", filename=str(log_file))
    logger_local.orange(f"Example 4 logs: {log_file}")

    # --- Build retriever with scores ------------------------------------
    retrieve_scored, vectorstore = rag_retrieval_with_scores(
        query="latest advancements in LLM compression 2025",
        search_count=5,
        min_score=0.15,
        cache_dir=f"{example_dir}/rag_cache",
        top_k=top_k,
    )

    # --- Test retrieval -------------------------------------------------
    sample_query = "What are the key techniques in LLM context compression from 2025?"
    scored_results: List[Tuple[Document, float]] = retrieve_scored(sample_query)

    # Save both content and scores
    scored_output = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in scored_results
    ]
    save_file(scored_output, f"{example_dir}/retrieved_with_scores.json")
    logger_local.success(f"Retrieved {len(scored_results)} docs (with scores) for: {sample_query}")

    # --- Build LangChain tool (now using vectorstore retriever) ---
    # Create proper retriever from vectorstore (not function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_web_context",
        "Search and return technical context from recent web sources on LLM compression.",
    )

    # --- Agent setup ----------------------------------------------------
    tools = [retriever_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    llm = ChatLlamaCpp(logger=logger_local, agent_name="rag_web_agent")
    llm_with_tools = llm.bind_tools(tools)

    def llm_call(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: MessagesState):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation: List[Document] = tool.invoke(tool_call["args"])
            content = "\n\n".join([d.page_content for d in observation])
            result.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool", END]:
        return "tool" if state["messages"][-1].tool_calls else END

    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm", llm_call)
    agent_builder.add_node("tool", tool_node)
    agent_builder.add_edge(START, "llm")
    agent_builder.add_conditional_edges("llm", should_continue, {"tool": "tool", END: END})
    agent_builder.add_edge("tool", "llm")
    agent = agent_builder.compile()

    # Visualize
    png_data = render_mermaid_graph(agent, xray=True, output_filename=f"{example_dir}/graph.png")
    display_iterm2_image(png_data)

    # Run agent
    user_query = "Summarize the latest LLM compression techniques from 2025 sources."
    result = agent.invoke({"messages": [HumanMessage(content=user_query)]})

    final_answer = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            final_answer = msg.content
            break

    result_clean = {
        "query": user_query,
        "retrieved_count": len(scored_results),
        "final_answer": final_answer,
    }
    (Path(example_dir) / "result.json").write_text(json.dumps(make_serializable(result_clean), indent=2))
    logger_local.success("RAG retrieval example completed.")
    logger_local.info(f"Final answer:\n{final_answer}")


if __name__ == "__main__":
    logger.magenta("Running all context engineering examples...")
    example_4_rag_retrieval()
    logger.green("All examples completed. Check generated/example_* folders.")