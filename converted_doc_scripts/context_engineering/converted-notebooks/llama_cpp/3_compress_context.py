# JetScripts/converted_doc_scripts/context_engineering/converted-notebooks/llama_cpp/3_compress_context.py
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.logger import logger
from jet.adapters.langchain.chat_agent_utils import compress_context

# -------------------------------------------------
# 1. Load & chunk documents
# -------------------------------------------------
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

# -------------------------------------------------
# 2. Embeddings & vector store (using jet modules)
# -------------------------------------------------
embeddings = EmbedLlamaCpp(model="embeddinggemma")
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

# -------------------------------------------------
# 3. LLM (local server, same as 2_select_context)
# -------------------------------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# -------------------------------------------------
# 4. Graph state & prompts
# -------------------------------------------------
from typing_extensions import Literal
from IPython.display import Image, display
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

class State(MessagesState):
    """Extended state that includes a summary field for context compression."""
    summary: str

rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng.
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

# -------------------------------------------------
# 5. Nodes
# -------------------------------------------------
def llm_call(state: MessagesState) -> dict:
    """Execute LLM call with system prompt and message history."""
    messages = [SystemMessage(content=rag_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node_with_compression(state: MessagesState) -> dict:
    """
    Fetch documents via the retriever tool and compress the retrieved text
    together with the conversation history using `compress_context`.
    """
    result = []
    messages = state["messages"]

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        # ---- compress retrieved docs + history ----
        compressed = compress_context(
            messages=messages,
            retriever_results=observation,
            max_tokens=3000,          # adjust as needed
            llm=llm
        )

        # optional token logging
        orig_tokens = count_tokens(observation, model="qwen3-instruct-2507:4b")
        comp_tokens = count_tokens(compressed, model="qwen3-instruct-2507:4b")
        logger.log(
            f"Context compressed: {orig_tokens} → {comp_tokens} tokens",
            colors=["INFO", "GREEN"]
        )

        result.append(ToolMessage(
            content=compressed,
            tool_call_id=tool_call["id"]
        ))
    return {"messages": result}

def summary_node(state: MessagesState) -> dict:
    """Generate a final summary after the loop ends."""
    summarization_prompt = """Summarize the full chat history and all tool feedback to
give an overview of what the user asked about and what the agent did."""
    messages = [SystemMessage(content=summarization_prompt)] + state["messages"]
    summary_msg = llm.invoke(messages)
    return {"summary": summary_msg.content}

# -------------------------------------------------
# 6. Conditional routing
# -------------------------------------------------
def should_continue(state: MessagesState) -> Literal["environment", "summary_node"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "environment"
    return "summary_node"

# -------------------------------------------------
# 7. Build the graph
# -------------------------------------------------
agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node_with_compression)
agent_builder.add_node("summary_node", summary_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "environment": "environment",
        "summary_node": "summary_node",
    },
)
agent_builder.add_edge("environment", "llm_call")
agent_builder.add_edge("summary_node", END)

agent = agent_builder.compile()
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# -------------------------------------------------
# 8. Example run
# -------------------------------------------------
query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": [HumanMessage(content=query)]})

# (Optional) pretty-print results – adapt `format_messages` if available
from rich.console import Console
from rich.markdown import Markdown
console = Console()
console.print("\n[bold magenta]Agent final messages:[/bold magenta]")
for msg in result["messages"]:
    console.print(msg)

if "summary" in result:
    console.print("\n[bold cyan]Conversation summary:[/bold cyan]")
    console.print(Markdown(result["summary"]))