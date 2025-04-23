from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from typing import Literal
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langchain_core.messages import RemoveMessage
from langgraph.graph import END
from langchain_core.messages import HumanMessage


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# How to delete messages

One of the common states for a graph is a list of messages. Usually you only add messages to that state. However, sometimes you may want to remove messages (either by directly modifying the state or as part of the graph). To do that, you can use the `RemoveMessage` modifier. In this guide, we will cover how to do that.

The key idea is that each state key has a `reducer` key. This key specifies how to combine updates to the state. The default `MessagesState` has a messages key, and the reducer for that key accepts these `RemoveMessage` modifiers. That reducer then uses these `RemoveMessage` to delete messages from the key.

So note that just because your graph state has a key that is a list of messages, it doesn't mean that that this `RemoveMessage` modifier will work. You also have to have a `reducer` defined that knows how to work with this.

**NOTE**: Many models expect certain rules around lists of messages. For example, some expect them to start with a `user` message, others expect all messages with tool calls to be followed by a tool message. **When deleting messages, you will want to make sure you don't violate these rules.**

## Setup

First, let's build a simple graph that uses messages. Note that it's using the `MessagesState` which has the required `reducer`.
"""

# %%capture --no-stderr
# %pip install --quiet -U langgraph jet.llm.ollama.base_langchain

"""
Next, we need to set API keys for Anthropic (the LLM we will use)
"""

# import getpass


# def _set_env(var: str):
# if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

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
    response = model.invoke(state["messages"])
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
    event["messages"][-1].pretty_logger.debug()


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_logger.debug()

"""
## Manually deleting messages

First, we will cover how to manually delete messages. Let's take a look at the current state of the thread:
"""

messages = app.get_state(config).values["messages"]
messages

"""
We can call `update_state` and pass in the id of the first message. This will delete that message.
"""


app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})

"""
If we now look at the messages, we can verify that the first one was deleted.
"""

messages = app.get_state(config).values["messages"]
messages

"""
## Programmatically deleting messages

We can also delete messages programmatically from inside the graph. Here we'll modify the graph to delete any old messages (longer than 3 messages ago) at the end of a graph run.
"""


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-3]]}


def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "delete_messages"
    return "action"


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_node(delete_messages)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("action", "agent")

workflow.add_edge("delete_messages", END)
app = workflow.compile(checkpointer=memory)

"""
We can now try this out. We can call the graph twice and then check the state
"""


config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug([(message.type, message.content)
                 for message in event["messages"]])


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug([(message.type, message.content)
                 for message in event["messages"]])

"""
If we now check the state, we should see that it is only three messages long. This is because we just deleted the earlier messages - otherwise it would be four!
"""

messages = app.get_state(config).values["messages"]
messages

"""
Remember, when deleting messages you will want to make sure that the remaining message list is still valid. This message list **may actually not be** - this is because it currently starts with an AI message, which some models do not allow.
"""

logger.info("\n\n[DONE]", bright=True)
