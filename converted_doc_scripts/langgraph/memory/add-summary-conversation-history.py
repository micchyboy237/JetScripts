from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from typing import Literal
from jet.adapters.langchain.chat_ollama import ChatOllama
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# How to add summary of the conversation history

One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. One way to work around that is to create a summary of the conversation to date, and use that with the past N messages. This guide will go through an example of how to do that.

This will involve a few steps:

- Check if the conversation is too long (can be done by checking number of messages or length of messages)
- If yes, the create summary (will need a prompt for this)
- Then remove all except the last N messages

A big part of this is deleting old messages. For an in depth guide on how to do that, see [this guide](../delete-messages)

## Setup

First, let's set up the packages we're going to want to use
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

## Build the chatbot

Let's now build the chatbot.
"""


memory = MemorySaver()


class State(MessagesState):
    summary: str


model = ChatOllama(model="llama3.1")


def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


workflow = StateGraph(State)

workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")

workflow.add_conditional_edges(
    "conversation",
    should_continue,
)

workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=memory)

"""
## Using the graph
"""


def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            logger.debug(m.content)
        if "summary" in v:
            logger.debug(v["summary"])


config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hi! I'm bob")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what's my name?")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the celtics!")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

"""
We can see that so far no summarization has happened - this is because there are only six messages in the list.
"""

values = app.get_state(config).values
values

"""
Now let's send another message in
"""

input_message = HumanMessage(content="i like how much they win")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

"""
If we check the state now, we can see that we have a summary of the conversation, as well as the last two messages
"""

values = app.get_state(config).values
logger.success(values)

"""
We can now resume having a conversation! Note that even though we only have the last two messages, we can still ask it questions about things mentioned earlier in the conversation (because we summarized those)
"""

input_message = HumanMessage(content="what's my name?")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what NFL team do you think I like?")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the patriots!")
logger.debug(input_message.content)
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

logger.info("\n\n[DONE]", bright=True)
