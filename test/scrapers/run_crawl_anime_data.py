from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # Or langchain-ollama
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.tools import Tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Tool for recursive crawling
def crawl_anime_data(url: str) -> str:
    loader = RecursiveUrlLoader(url=url, max_depth=2)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

tools = [Tool(name="crawl_anime", func=crawl_anime_data, description="Recursively crawl anime site")]

# LLM and prompt
llm = ChatOpenAI(temperature=0)  # Or local LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a ReAct agent for finding top 10 Isekai anime 2025 (release, synopsis, episodes, status).
    Crawl data, extract fields, and check: Are 10 entries complete? If not, crawl new URLs or refine. Return JSON.""" ),
    MessagesPlaceholder(variable_name="messages"),
])

# Graph state
class GraphState(MessagesState):
    complete: bool = False

# Nodes
def agent(state: GraphState):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def reviewer(state: GraphState):
    last_output = state["messages"][-1].content
    # Simple check: count valid entries
    valid_entries = last_output.count('"title":') if "title" in last_output else 0
    state.complete = valid_entries >= 10 and "synopsis" in last_output
    return {"messages": [HumanMessage(content=f"Found {valid_entries}/10 entries")], "complete": state.complete}

# Graph
workflow = StateGraph(GraphState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reviewer", reviewer)
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "end": "reviewer"})
workflow.add_conditional_edges("reviewer", lambda s: "agent" if not s["complete"] else END)
workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")

# Run
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "anime_crawl"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="Top 10 Isekai 2025: release, synopsis, eps, status")]},
    config
)
print(result["messages"][-1].content)
