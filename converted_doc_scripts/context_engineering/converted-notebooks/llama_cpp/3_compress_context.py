import json
from pathlib import Path
import shutil
from typing import Optional, List, Any
from datetime import datetime

from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.logger import logger, CustomLogger
from jet.visualization.terminal import display_iterm2_image
from jet.search.searxng import search_searxng, SearchResult
import os

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
os.makedirs(str(BASE_OUTPUT_DIR), exist_ok=True)
log_file = os.path.join(str(BASE_OUTPUT_DIR), "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")

from jet.adapters.langchain.chat_agent_utils import compress_context

DEFAULT_QUERY_URL = "http://jethros-macbook-air.local:8888"

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split large documents into smaller chunks suitable for embedding.

    Args:
        documents: List of loaded documents
        chunk_size: Maximum token length per chunk (safe under 2048)
        chunk_overlap: Overlap between chunks to preserve context

    Returns:
        List of chunked Document objects with preserved metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Approximate; actual token count handled by server
        add_start_index=True,
    )
    return splitter.split_documents(documents)

# ----------------------------------------------------------------------
# 1. Search helper – reusable, generic, uses all SearXNG result fields
# ----------------------------------------------------------------------
def retrieve_relevant_urls(
    query: str,
    count: int = 5,
    min_score: float = 0.1,
    min_date: Optional[datetime] = None,
    include_sites: Optional[List[str]] = None,
    exclude_sites: Optional[List[str]] = None,
) -> List[str]:
    """
    Retrieve and rank relevant URLs using SearXNG results.

    - Calls ``search_searxng`` (default instance URL from config or env)
    - Filters out malformed / short-content results
    - Ranks by:
        * base ``score`` (SearXNG relevance)
        * recency boost (if ``publishedDate`` is parseable)
        * content-length boost (log-scaled)
    - Returns only the top ``count`` URLs.
    """
    # Fetch more than needed – we will filter & rank
    raw_results: List[SearchResult] = search_searxng(
        query_url=DEFAULT_QUERY_URL,               # uses default from module / Redis config
        query=query,
        # count=count * 3,
        min_score=min_score,
        min_date=min_date,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
        use_cache=True,
    )
    save_file(raw_results, f"{str(BASE_OUTPUT_DIR)}/web_all_searxng_results.json")

    # ------------------------------------------------------------------
    # Filter valid, useful results
    # ------------------------------------------------------------------
    filtered: List[SearchResult] = []
    for res in raw_results:
        if not (res.get("url") and res.get("title") and res.get("content")):
            continue
        if len(res["content"]) < 50:                     # skip tiny snippets
            continue
        filtered.append(res)
    save_file(filtered, f"{str(BASE_OUTPUT_DIR)}/web_filtered_searxng_results.json")

    # ------------------------------------------------------------------
    # Composite ranking function
    # ------------------------------------------------------------------
    now = datetime.now()

    def composite_rank(res: SearchResult) -> float:
        base = res.get("score", 0.0)

        # Recency boost
        recency = 0.0
        pub_str = res.get("publishedDate", "")
        if pub_str:
            try:
                pub_date = datetime.fromisoformat(pub_str.split("T")[0])
                days_old = (now - pub_date).days
                recency = max(0.0, 1.0 - days_old / 365.0) * 0.3
            except Exception:
                pass

        # Content length boost (capped)
        length_boost = min(len(res["content"]) / 1000.0, 1.0) * 0.1

        return base + recency + length_boost

    filtered.sort(key=composite_rank, reverse=True)
    save_file(filtered, f"{str(BASE_OUTPUT_DIR)}/web_sorted_searxng_results.json")

    # Return only URLs
    return [res["url"] for res in filtered[:count]]


# ----------------------------------------------------------------------
# 2. Load & chunk documents (now driven from search results)
# ----------------------------------------------------------------------
def load_documents_from_urls(urls: List[str]) -> List[Any]:
    """Wrapper around WebBaseLoader that returns a flat list of Document objects."""
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


# ----------------------------------------------------------------------
# 3. Example function – all user-customizable variables are here
# ----------------------------------------------------------------------
def example_1_rag_with_compression_and_summary(
    # ------------------- USER CONFIG -------------------
    query: str = "latest advancements in LLM compression 2025",
    search_count: int = 5,
    min_score: float = 0.15,
    system_prompt: str = (
        "You are a research assistant. Summarize and compress context from web sources. "
        "Focus on technical accuracy and novelty. Output in bullet points."
    ),
    user_prompt_template: str = (
        "Compress and summarize the following documents into a concise technical overview:\n\n{documents}"
    ),
    # --------------------------------------------------
) -> None:
    """
    Full RAG + compression + summarization pipeline.

    All customizable values (query, prompts, search params) live **only** in this function.
    """
    example_dir = Path(BASE_OUTPUT_DIR) / "example_1_rag_with_compression_and_summary"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger_local = CustomLogger("example_1_rag_with_compression_and_summary", filename=str(log_file))
    logger_local.orange(f"Example 1 logs: {log_file}")

    # -------------------------------------------------
    # 1. Retrieve relevant URLs
    # -------------------------------------------------
    logger_local.info(f"Searching SearXNG for: {query}")
    urls = retrieve_relevant_urls(
        query=query,
        count=search_count,
        min_score=min_score,
    )
    logger_local.success(f"Retrieved {len(urls)} URLs: {urls}")

    # -------------------------------------------------
    # 2. Load & chunk the retrieved pages
    # -------------------------------------------------
    docs_list = load_documents_from_urls(urls)
    save_file(docs_list, f"{str(example_dir)}/docs.json")

    # Split large documents to avoid embedding token limit
    chunked_docs = chunk_documents(docs_list, chunk_size=1500, chunk_overlap=200)
    save_file(chunked_docs, f"{str(example_dir)}/chunks.json")

    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=2000, chunk_overlap=50
    # )
    # doc_splits = text_splitter.split_documents(docs_list)

    # -------------------------------------------------
    # 3. Embeddings & vector store
    # -------------------------------------------------
    embeddings = EmbedLlamaCpp(model="embeddinggemma")
    vectorstore = InMemoryVectorStore.from_documents(
        # documents=doc_splits,
        documents=chunked_docs,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_web_pages",
        "Search and return information from the retrieved web pages.",
    )

    # -------------------------------------------------
    # 4. LLM (local server)
    # -------------------------------------------------
    llm = ChatLlamaCpp(logger=logger_local)

    tools = [retriever_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    logger_local.info(f"Tools ({len(tools)})")
    logger_local.debug(format_json(list(tools_by_name.keys())))

    # -------------------------------------------------
    # 5. Graph state & prompts
    # -------------------------------------------------
    from typing_extensions import Literal
    from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    class State(MessagesState):
        summary: str

    # Use the **user-provided** system prompt for the RAG agent
    rag_prompt = system_prompt

    # -------------------------------------------------
    # 6. Nodes (unchanged except they now receive the generic prompt)
    # -------------------------------------------------
    def llm_call(state: MessagesState) -> dict:
        messages = [SystemMessage(content=rag_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node_with_compression(state: MessagesState) -> dict:
        result = []

        for tool_call in state["messages"][-1].tool_calls:
            logger_local.info("Invoking Tool Call - observation:\n%s", format_json(tool_call))
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

            logger_local.info("Tools Result - observation")
            logger_local.debug(observation)

            # Filter out ToolMessage messages with empty content
            messages = [
                msg for msg in state["messages"]
                if not (isinstance(msg, ToolMessage) and (msg.content is None or msg.content == ""))
            ]

            messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
            logger_local.info("Invoking compress_context:\n%s", format_json({
                "max_tokens": 4096,
                "messages": messages,
                "retriever_results": observation,
            }))
            compressed = compress_context(
                messages=messages,
                retriever_results=observation,
                max_tokens=4096,
                llm=llm,
                logger=logger_local,
            )

            logger_local.info("Compression Result - observation")
            logger_local.debug(compressed)

            orig_tokens = count_tokens(observation, model="qwen3-instruct-2507:4b")
            comp_tokens = count_tokens(compressed, model="qwen3-instruct-2507:4b")
            logger_local.log(
                f"Context compressed: {orig_tokens} → {comp_tokens} tokens",
                colors=["INFO", "GREEN"],
            )

            result.append(
                ToolMessage(content=compressed, tool_call_id=tool_call["id"])
            )
        return {"messages": result}

    def summary_node(state: MessagesState) -> dict:
        history_lines = []
        for msg in state["messages"]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            if isinstance(msg, ToolMessage):
                history_lines.append(f"Retrieved context: {msg.content}")
            else:
                history_lines.append(f"{role}: {msg.content}")
        history_text = "\n".join(history_lines)

        summarization_system = """You are a concise summarizer.  
Produce a **single paragraph** that captures:
- the user's exact research question,
- the key retrieved facts / techniques,
- the final answer or conclusion.
Preserve all technical terms, citations, and numbers.  
Do **not** add speculation or extra commentary."""

        summary_msg = llm.invoke(
            [
                SystemMessage(content=summarization_system),
                HumanMessage(content=history_text),
            ]
        )
        logger_local.info("Summary Node Result:")
        logger_local.debug(format_json(summary_msg))
        return {"summary": summary_msg.content}

    # -------------------------------------------------
    # 7. Conditional routing
    # -------------------------------------------------
    def should_continue(state: MessagesState) -> Literal["environment", "summary_node"]:
        last_message = state["messages"][-1]
        return "environment" if last_message.tool_calls else "summary_node"

    # -------------------------------------------------
    # 8. Build the graph
    # -------------------------------------------------
    agent_builder = StateGraph(State)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node_with_compression)
    agent_builder.add_node("summary_node", summary_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"environment": "environment", "summary_node": "summary_node"},
    )
    agent_builder.add_edge("environment", "llm_call")
    agent_builder.add_edge("summary_node", END)

    agent = agent_builder.compile()

    # -------------------------------------------------
    # 9. Visualise & run
    # -------------------------------------------------
    png_path = example_dir / "agent_graph.png"
    png_data = render_mermaid_graph(agent, xray=True, output_filename=str(png_path))
    display_iterm2_image(png_data)

    # Build the final user message using the template
    user_message = user_prompt_template.format(documents="\n\n".join(urls))
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    logger_local.info("Final Message Result:")
    logger_local.debug(format_json(result))

    # -------------------------------------------------
    # 10. Save outputs
    # -------------------------------------------------
    result_clean = {
        "messages": [
            m.model_dump() if hasattr(m, "dict") else str(m) for m in result["messages"]
        ],
        "summary": result.get("summary", ""),
    }
    (example_dir / "result.json").write_text(json.dumps(result_clean, indent=2))

    if "summary" in result:
        (example_dir / "summary.md").write_text(result["summary"])

    logger_local.magenta("\nExample 1 - Final Messages:")
    for msg in result["messages"]:
        logger_local.debug(msg)
    if "summary" in result:
        logger_local.info("\nExample 1 - Conversation Summary:")
        logger_local.success(result["summary"])


# ----------------------------------------------------------------------
# Entry point – expose the example with sensible defaults
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logger.magenta("Running 3_compress_context.py example...")
    example_1_rag_with_compression_and_summary()
    logger.green("Example completed.")