# === REFACTOR: Per-example output directories and modular functions ===

import json
import logging
from pathlib import Path

import shutil
from typing import TypedDict
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from rich.pretty import pprint
from jet.visualization.terminal import display_iterm2_image
import os
from jet.logger import logger

from langgraph.graph import END, START, StateGraph

from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from jet.models.utils import get_embedding_size
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp

import math
import types
import inspect
import pydantic
from langchain_core.tools import StructuredTool
from typing import Any
from tqdm import tqdm
import uuid
from langgraph.store.base import PutOp, BaseStore
from langgraph.checkpoint.memory import InMemorySaver

from jet.adapters.langchain.chat_agent_utils import build_agent

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from typing_extensions import Literal

from jet.adapters.llama_cpp.tokens import count_tokens

# ------ Global objects
BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
embeddings = EmbedLlamaCpp(model="embeddinggemma")
llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

# -------- Tool convert from math functions
def safe_tool_from_function(func) -> StructuredTool | None:
    """Create StructuredTool from math builtin, safely handling missing signatures."""
    if not isinstance(func, types.BuiltinFunctionType):
        return None
    try:
        sig = inspect.signature(func)
    except ValueError:
        return None

    fields: dict[str, tuple[Any, Any]] = {}
    for param in sig.parameters.values():
        name = param.name
        anno = param.annotation if param.annotation is not param.empty else float

        if param.default is param.empty:
            fields[name] = (anno, ...)
        else:
            default = param.default
            if default is None:
                default = None
            elif isinstance(default, (int, str, bool)):
                default = default
            elif isinstance(default, float):
                default = float(default)
            else:
                default = None
            fields[name] = (anno, pydantic.Field(default=default))
    try:
        ArgsSchema = pydantic.create_model(
            f"{func.__name__.capitalize()}Args",
            **fields
        )
    except Exception as e:
        print(f"[WARN] Failed to create schema for {func.__name__}: {e}")
        return None

    def wrapper(**kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        args_list = []
        kwargs_dict = {}

        for param in sig.parameters.values():
            name = param.name
            kind = param.kind

            if kind is inspect.Parameter.POSITIONAL_ONLY:
                if name in clean_kwargs:
                    args_list.append(clean_kwargs.pop(name))
                elif param.default is not param.empty:
                    args_list.append(param.default)
                else:
                    raise TypeError(f"Missing required positional-only argument: {name}")

            elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if name in clean_kwargs:
                    args_list.append(clean_kwargs.pop(name))
                elif param.default is not param.empty:
                    args_list.append(param.default)
                else:
                    raise TypeError(f"Missing required argument: {name}")

            elif kind is inspect.Parameter.VAR_POSITIONAL:
                if name in clean_kwargs:
                    var_val = clean_kwargs.pop(name)
                    if not isinstance(var_val, (list, tuple)):
                        raise TypeError(f"VAR_POSITIONAL param '{name}' must be a list/tuple")
                    args_list.extend(var_val)

            elif kind is inspect.Parameter.KEYWORD_ONLY:
                if name in clean_kwargs:
                    kwargs_dict[name] = clean_kwargs.pop(name)
                elif param.default is not param.empty:
                    kwargs_dict[name] = param.default
                else:
                    raise TypeError(f"Missing required keyword-only argument: {name}")

            elif kind is inspect.Parameter.VAR_KEYWORD:
                pass

        accepts_varkw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if clean_kwargs:
            if accepts_varkw:
                kwargs_dict.update(clean_kwargs)
            else:
                unexpected = ", ".join(clean_kwargs.keys())
                raise TypeError(f"Got unexpected keyword arguments: {unexpected}")

        return func(*args_list, **kwargs_dict)

    return StructuredTool(
        name=func.__name__,
        description=getattr(func, "__doc__", "") or f"Call {func.__name__}",
        args_schema=ArgsSchema,
        func=wrapper,
    )


# ==== EXAMPLES ====

def example_1_basic_joke():
    """Example 1: Basic joke generation and improvement with graph visualization."""
    example_dir = os.path.join(BASE_OUTPUT_DIR, "example_1_basic_joke")
    os.makedirs(example_dir, exist_ok=True)
    log_file = f"{example_dir}/main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 1 logs: {log_file}")
    
    class State(TypedDict):
        topic: str
        joke: str
        improved_joke: str

    def generate_joke(state: State) -> dict[str, str]:
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}

    def improve_joke(state: State) -> dict[str, str]:
        print(f"Initial joke: {state['joke']}")
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}

    workflow = StateGraph(State)
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", "improve_joke")
    workflow.add_edge("improve_joke", END)
    chain = workflow.compile()

    png_data = render_mermaid_graph(
        chain, output_filename=f"{example_dir}/joke_generator_graph.png"
    )
    display_iterm2_image(png_data)

    result = chain.invoke({"topic": "quantum physics"})
    # Save final result
    (Path(example_dir) / "result.json").write_text(json.dumps(result, indent=2))
    logger.green("\nExample 1 Result:")
    pprint(result)


def example_2_memory_aware_joke():
    """Example 2: Joke generation with in-memory context to avoid repetition."""
    example_dir = os.path.join(BASE_OUTPUT_DIR, "example_2_memory_aware_joke")
    os.makedirs(example_dir, exist_ok=True)
    log_file = f"{example_dir}/main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 2 logs: {log_file}")
    
    class State(TypedDict):
        topic: str
        joke: str

    namespace = ("rlm", "joke_generator")
    checkpointer = InMemorySaver()
    memory_store = InMemoryStore(
        index={"embed": embeddings, "dims": get_embedding_size("embeddinggemma")}
    )

    def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
        # Use injected `store` (which is `memory_store`)
        prior_joke = store.get(namespace, "last_joke")
        prior_text = prior_joke.value["joke"] if prior_joke else "None"
        print(f"Prior joke: {prior_text}")
        prompt = (
            f"Write a short joke about {state['topic']}, "
            f"different from: {prior_text}"
        )
        msg = llm.invoke(prompt)
        store.put(namespace, "last_joke", {"joke": msg.content})
        return {"joke": msg.content}

    workflow = StateGraph(State)
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", END)
    chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

    png_data = render_mermaid_graph(
        chain, output_filename=f"{example_dir}/memory_joke_graph.png"
    )
    display_iterm2_image(png_data)

    # Run twice to show memory effect
    logger.cyan("\nFirst run:")
    result1 = chain.invoke({"topic": "AI"}, config={"configurable": {"thread_id": "joke_thread"}})
    (Path(example_dir) / "run1_result.json").write_text(json.dumps(result1, indent=2))
    logger.cyan(str(result1))

    logger.cyan("\nSecond run (should be different):")
    result2 = chain.invoke({"topic": "AI"}, config={"configurable": {"thread_id": "joke_thread"}})
    (Path(example_dir) / "run2_result.json").write_text(json.dumps(result2, indent=2))
    logger.cyan(str(result2))


def example_3_structured_tools():
    """Example 3: Auto-generate StructuredTool from math functions and run agent."""
    from typing import TypedDict, List
    from langchain_core.messages import BaseMessage

    class AgentState(TypedDict):
        """State for the tool-selection workflow."""
        messages: List[BaseMessage]          # always present
        selected_tools: List[StructuredTool] # optional, filled by select_tools

    example_dir = os.path.join(BASE_OUTPUT_DIR, "example_3_structured_tools")
    os.makedirs(example_dir, exist_ok=True)
    log_file = f"{example_dir}/main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 3 logs: {log_file}")

    all_tools = []
    for function_name in tqdm(dir(math), desc="Building tools"):
        func = getattr(math, function_name)
        if isinstance(func, types.BuiltinFunctionType):
            tool = safe_tool_from_function(func)
            if tool:
                all_tools.append(tool)
    all_tools = all_tools[:30]

    logger.purple("\nAll Tools:")
    pprint(all_tools)
    # === SAVE TOOLS LIST ===
    tools_info = [{"name": t.name, "description": t.description} for t in all_tools]
    (Path(example_dir) / "tools.json").write_text(json.dumps(tools_info, indent=2))

    tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}
    # === INDEX TOOL DESCRIPTIONS IN VECTOR STORE ===
    tool_store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": get_embedding_size("embeddinggemma"),
            "fields": ["description"],
        }
    )

    put_ops = []
    for tool_id, tool in tool_registry.items():
        put_ops.append(
            PutOp(
                namespace=("tools",),
                key=tool_id,
                value={"description": f"{tool.name}: {tool.description}"},
            )
        )
    tool_store.batch(put_ops)

    # Save indexed tools
    indexed_tools = [
        {"id": tid, "name": tool_registry[tid].name, "description": tool_registry[tid].description}
        for tid in tool_registry
    ]
    (Path(example_dir) / "indexed_tools.json").write_text(json.dumps(indexed_tools, indent=2))

    # === DYNAMIC TOOL SELECTION AGENT ===
    def select_relevant_tools(state: AgentState) -> AgentState:
        query = state["messages"][-1].content
        # `search` signature:  namespace_prefix (positional) /  *, query, limit, …
        all_results = tool_store.search(
            ("tools",),      # positional namespace_prefix
            query=query,
            limit=20
        )
        
        # No need to filter by namespace — already scoped
        results = [item for item in all_results if item.key in tool_registry][:5]

        if not results:
            logger.yellow("No relevant tools found. Using fallback.")
            selected_tools = all_tools[:3]
        else:
            selected_tool_ids = [item.key for item in results]
            selected_tools = [tool_registry[tid] for tid in selected_tool_ids]

        # Save selection
        (Path(example_dir) / "retrieved_tool_ids.json").write_text(
            json.dumps([item.key for item in all_results], indent=2)
        )
        (Path(example_dir) / "selected_tool_ids.json").write_text(
            json.dumps([item.key for item in results], indent=2)
        )
        (Path(example_dir) / "selected_tool_names.json").write_text(
            json.dumps([t.name for t in selected_tools], indent=2)
        )

        # Preserve the incoming messages for the next node
        return {"selected_tools": selected_tools, "messages": state["messages"]}

    # Build agent with dynamic tools
    workflow = StateGraph(AgentState)
    workflow.add_node("select_tools", select_relevant_tools)
    workflow.add_node("agent", lambda state: build_agent(
        tools=state.get("selected_tools", []),
        model=llm
    ).invoke({"messages": state["messages"]}))
    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "agent")
    workflow.add_edge("agent", END)
    dynamic_agent = workflow.compile()

    png_data = render_mermaid_graph(
        dynamic_agent, output_filename=f"{example_dir}/dynamic_tool_selection_graph.png"
    )
    display_iterm2_image(png_data)

    query = "Use available tools to calculate arc cosine of 0.5."
    result = dynamic_agent.invoke({"messages": [HumanMessage(content=query)]})

    # Save final result
    result_clean = {
        "query": query,
        "selected_tools": [t.name for t in result.get("selected_tools", [])],
        "final_answer": result.get("messages", [-1])[-1].content if result.get("messages") else ""
    }
    (Path(example_dir) / "agent_result.json").write_text(json.dumps(result_clean, indent=2))

    logger.purple("\nAgent tool result:")
    pprint(result)


def example_4_rag_retrieval():
    """Example 4: RAG agent with token-budgeted retrieval from Lilian Weng blogs."""
    example_dir = os.path.join(BASE_OUTPUT_DIR, "example_4_rag_retrieval")
    os.makedirs(example_dir, exist_ok=True)
    log_file = f"{example_dir}/main.log"
    logger.basicConfig(filename=log_file, level=logging.INFO, force=True)
    logger.orange(f"Example 4 logs: {log_file}")
    
    urls = [
        "https://lilianweng.github.io/posts/2025-05-01-thinking/",
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(documents=doc_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever, "retrieve_blog_posts", "Search and return information about Lilian Weng blog posts."
    )
    tools = [retriever_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng.
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

    def llm_call(state: MessagesState):
        messages = state["messages"]
        user_query = messages[-1].content
        retrieved = retriever.invoke(user_query)
        doc_texts = [doc.page_content for doc in retrieved[:4]]
        doc_token_counts = count_tokens(doc_texts, model="qwen3-instruct-2507:4b", prevent_total=True, add_special_tokens=False)

        history_lines = [
            f"{ 'User' if isinstance(m, HumanMessage) else 'Assistant' }: {m.content}"
            for m in messages[:-1]
        ]
        history_text = "\n".join(history_lines)

        MAX_CTX = 3000
        SAFETY_BUFFER = 600
        BUDGET = MAX_CTX - SAFETY_BUFFER
        static_parts = rag_prompt + "\n\nContext:\nDocuments:\n\nHistory:\n" + history_text + "\n\nQuestion: " + user_query
        static_tokens = count_tokens(static_parts, model="qwen3-instruct-2507:4b")
        available_for_docs = BUDGET - static_tokens - 200

        selected_docs = []
        consumed = 0
        for txt, cnt in zip(doc_texts, doc_token_counts):
            if consumed + cnt <= available_for_docs:
                selected_docs.append(txt)
                consumed += cnt
            else:
                break

        doc_text = "\n\n".join(selected_docs) if selected_docs else ""
        final_context = f"Documents:\n{doc_text}\n\nHistory:\n{history_text}"
        final_prompt = f"{rag_prompt}\n\nContext:\n{final_context}\nQuestion: {user_query}"
        total_tokens = count_tokens(final_prompt, model="qwen3-instruct-2507:4b")
        logger.log("final_prompt_tokens: ", total_tokens, colors=["INFO", "DEBUG"])

        response = llm_with_tools.invoke([SystemMessage(content=final_prompt)])
        return {"messages": [response]}

    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["environment", END]:
        last_message = state["messages"][-1]
        return "Action" if last_message.tool_calls else END

    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
    agent_builder.add_edge("environment", "llm_call")
    agent = agent_builder.compile()

    png_data = render_mermaid_graph(agent, xray=True, output_filename=f"{example_dir}/blog_retrieval_graph.png")
    display_iterm2_image(png_data)

    query = "What are the types of reward hacking discussed in the blogs?"
    result = agent.invoke({"messages": query})
    
    # === SAVE RAG RESULT ===
    result_clean = {
        "query": query,
        "messages": [
            m.dict() if hasattr(m, "dict") else {"type": type(m).__name__, "content": str(m)}
            for m in result.get("messages", [])
        ],
        "final_answer": next(
            (m.content for m in reversed(result.get("messages", [])) if hasattr(m, "content")),
            ""
        )
    }
    (Path(example_dir) / "rag_result.json").write_text(json.dumps(result_clean, indent=2))

    logger.purple("\nAgent query result:")
    pprint(result)


# === ADD MAIN BLOCK ===
if __name__ == "__main__":
    logger.magenta("Running all context engineering examples...")
    example_1_basic_joke()
    example_2_memory_aware_joke()
    example_3_structured_tools()
    example_4_rag_retrieval()
    logger.green("All examples completed. Check generated/example_* folders.")

