from langchain_core.messages import HumanMessage
from IPython.display import display, Image
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import MessagesState
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.runnables import ConfigurableField
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

"""
# How to stream from the final node
"""

"""
<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>            
                <a href="https://langchain-ai.github.io/langgraph/concepts/streaming/">
                    Streaming
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#chat-models/">
                    Chat Models
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#tools">
                    Tools
                </a>
            </li>
        </ul>
    </p>
</div> 

A common use case when streaming from an agent is to stream LLM tokens from inside the final node. This guide demonstrates how you can do this.

## Setup

First let's install our required packages and set our API keys
"""

# %%capture --no-stderr
# %pip install -U langgraph langchain-openai langchain-community

# import getpass


def _set_env(var: str):
    if not os.environ.get(var):
        #         os.environ[var] = getpass.getpass(f"{var}: ")

        # _set_env("OPENAI_API_KEY")


"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>
"""

"""
## Define model and tools
"""


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOllama(model="llama3.1")
final_model = ChatOllama(model="llama3.1")

model = model.bind_tools(tools)
final_model = final_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

"""
## Define graph
"""


def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "final"


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_final_model(state: MessagesState):
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = final_model.invoke(
        [
            SystemMessage("Rewrite this in the voice of Al Roker"),
            HumanMessage(last_ai_message.content),
        ]
    )
    response.id = last_ai_message.id
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")
builder.add_edge("final", END)

graph = builder.compile()


display(Image(graph.get_graph().draw_mermaid_png()))

"""
## Stream outputs from the final node
"""

"""
### Filter on event metadata
"""

"""
First option to get the LLM events from within a specific node (`final` node in our case) is to filter on the `langgraph_node` field in the event metadata. This will be sufficient in case you need to stream events from ALL LLM calls inside the node. This means that if you have multiple different LLMs invoked inside the node, this filter will include events from all of them.
"""


inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata["langgraph_node"] == "final"
    ):
        print(msg.content, end="|", flush=True)

"""
### Filter on custom tags
"""

"""
Alternatively, you can add configuration with custom tags to your LLM, like we did in the beginning, by adding `final_model.with_config(tags=["final_node"])`. This will allow us to more precisely filter the events to keep the ones only from this model.
"""

inputs = {"messages": [HumanMessage(content="what's the weather in nyc?")]}
for event in graph.stream_events(inputs, version="v2"):
    kind = event["event"]
    tags = event.get("tags", [])
    if kind == "on_chat_model_stream" and "final_node" in event.get("tags", []):
        data = event["data"]
        if data["chunk"].content:
            print(data["chunk"].content, end="|", flush=True)

logger.info("\n\n[DONE]", bright=True)
