from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from typing import Literal
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from typing import Literal
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

initialize_ollama_settings()

"""
# How to manage conversation history

One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. In order to prevent this from happening, you need to probably manage the conversation history.

Note: this guide focuses on how to do this in LangGraph, where you can fully customize how this is done. If you want a more off-the-shelf solution, you can look into functionality provided in LangChain:

- [How to filter messages](https://python.langchain.com/docs/how_to/filter_messages/)
- [How to trim messages](https://python.langchain.com/docs/how_to/trim_messages/)
"""

"""
## Setup

First, let's set up the packages we're going to want to use
"""

# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain_ollama

"""
Next, we need to set API keys for Anthropic (the LLM we will use)
"""

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>
"""

"""
## Build the agent
Let's now build a simple ReAct style agent.
"""


memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatOllama(model="llama3.1")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"


def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    return {"messages": response}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["action", END],
)

workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

"""
## Filtering messages

The most straight-forward thing to do to prevent conversation history from blowing up is to filter the list of messages before they get passed to the LLM. This involves two parts: defining a function to filter messages, and then adding it to the graph. See the example below which defines a really simple `filter_messages` function and then uses it.
"""


memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatOllama(model="llama3.1")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"


def filter_messages(messages: list):
    return messages[-1:]


def call_model(state: MessagesState):
    messages = filter_messages(state["messages"])
    response = bound_model.invoke(messages)
    return {"messages": response}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["action", END],
)

workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

"""
In the above example we defined the `filter_messages` function ourselves. We also provide off-the-shelf ways to trim and filter messages in LangChain. 

- [How to filter messages](https://python.langchain.com/docs/how_to/filter_messages/)
- [How to trim messages](https://python.langchain.com/docs/how_to/trim_messages/)
"""

logger.info("\n\n[DONE]", bright=True)
