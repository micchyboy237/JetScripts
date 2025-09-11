from IPython.display import display, Image
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import RemoveMessage
from langchain_core.messages.utils import (
    # highlight-next-line
    trim_messages,
    # highlight-next-line
    count_tokens_approximately
    # highlight-next-line
)
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode
from typing import Any
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# How to manage conversation history in a ReAct Agent

!!! info "Prerequisites"
    This guide assumes familiarity with the following:

    - [Prebuilt create_react_agent](../create-react-agent)
    - [Persistence](../../concepts/persistence)
    - [Short-term Memory](../../concepts/memory/#short-term-memory)
    - [Trimming Messages](https://python.langchain.com/docs/how_to/trim_messages/)

Message history can grow quickly and exceed LLM context window size, whether you're building chatbots with many conversation turns or agentic systems with numerous tool calls. There are several strategies for managing the message history:

* [message trimming](#keep-the-original-message-history-unmodified) — remove first or last N messages in the history
* [summarization](#summarizing-message-history) — summarize earlier messages in the history and replace them with a summary
* custom strategies (e.g., message filtering, etc.)

To manage message history in `create_react_agent`, you need to define a `pre_model_hook` function or [runnable](https://python.langchain.com/docs/concepts/runnables/) that takes graph state an returns a state update:


* Trimming example:
    ```python
    # highlight-next-line
    
    # This function will be called every time before the node that calls LLM
    def pre_model_hook(state):
        trimmed_messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=384,
            start_on="human",
            end_on=("human", "tool"),
        )
        # You can return updated messages either under `llm_input_messages` or 
        # `messages` key (see the note below)
        # highlight-next-line
        return {"llm_input_messages": trimmed_messages}

    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model,
        tools,
        # highlight-next-line
        pre_model_hook=pre_model_hook,
        checkpointer=checkpointer,
    )
    ```

* Summarization example:
    ```python
    # highlight-next-line
    
    model = ChatOllama(model="llama3.2")
    
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=model,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
    )

    class State(AgentState):
        # NOTE: we're adding this key to keep track of previous summary information
        # to make sure we're not summarizing on every LLM call
        # highlight-next-line
        context: dict[str, Any]
    
    
    checkpointer = InMemorySaver()
    graph = create_react_agent(
        model,
        tools,
        # highlight-next-line
        pre_model_hook=summarization_node,
        # highlight-next-line
        state_schema=State,
        checkpointer=checkpointer,
    )
    ```

!!! Important
    
    * To **keep the original message history unmodified** in the graph state and pass the updated history **only as the input to the LLM**, return updated messages under `llm_input_messages` key
    * To **overwrite the original message history** in the graph state with the updated history, return updated messages under `messages` key
    
    To overwrite the `messages` key, you need to do the following:

    ```python

    def pre_model_hook(state):
        updated_messages = ...
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *updated_messages]
            ...
        }
    ```

## Setup

First, let's install the required packages and set our API keys
"""
logger.info("# How to manage conversation history in a ReAct Agent")

# %%capture --no-stderr
# %pip install -U langgraph langchain-ollama langmem

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("OPENAI_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Keep the original message history unmodified

Let's build a ReAct agent with a step that manages the conversation history: when the length of the history exceeds a specified number of tokens, we will call [`trim_messages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) utility that that will reduce the history while satisfying LLM provider constraints.

There are two ways that the updated message history can be applied inside ReAct agent:

  * [**Keep the original message history unmodified**](#keep-the-original-message-history-unmodified) in the graph state and pass the updated history **only as the input to the LLM**
  * [**Overwrite the original message history**](#overwrite-the-original-message-history) in the graph state with the updated history

Let's start by implementing the first one. We'll need to first define model and tools for our agent:
"""
logger.info("## Keep the original message history unmodified")


model = ChatOllama(model="llama3.2")


def get_weather(location: str) -> str:
    """Use this to get weather information."""
    if any([city in location.lower() for city in ["nyc", "new york city"]]):
        return "It might be cloudy in nyc, with a chance of rain and temperatures up to 80 degrees."
    elif any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's always sunny in sf"
    else:
        return f"I am not sure what the weather is in {location}"


tools = [get_weather]

"""
Now let's implement `pre_model_hook` — a function that will be added as a new node and called every time **before** the node that calls the LLM (the `agent` node).

Our implementation will wrap the `trim_messages` call and return the trimmed messages under `llm_input_messages`. This will **keep the original message history unmodified** in the graph state and pass the updated history **only as the input to the LLM**
"""
logger.info("Now let's implement `pre_model_hook` — a function that will be added as a new node and called every time **before** the node that calls the LLM (the `agent` node).")


def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}


checkpointer = InMemorySaver()
graph = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)


display(Image(graph.get_graph().draw_mermaid_png()))

"""
We'll also define a utility to render the agent outputs nicely:
"""
logger.info("We'll also define a utility to render the agent outputs nicely:")


def print_stream(stream, output_messages_key="llm_input_messages"):
    for chunk in stream:
        for node, update in chunk.items():
            logger.debug(f"Update from node: {node}")
            messages_key = (
                output_messages_key if node == "pre_model_hook" else "messages"
            )
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    logger.debug(message)
                else:
                    message.pretty_logger.debug()

        logger.debug("\n\n")


"""
Now let's run the agent with a few different queries to reach the specified max tokens limit:
"""
logger.info(
    "Now let's run the agent with a few different queries to reach the specified max tokens limit:")

config = {"configurable": {"thread_id": "1"}}

inputs = {"messages": [("user", "What's the weather in NYC?")]}
result = graph.invoke(inputs, config=config)

inputs = {"messages": [("user", "What's it known for?")]}
result = graph.invoke(inputs, config=config)

"""
Let's see how many tokens we have in the message history so far:
"""
logger.info("Let's see how many tokens we have in the message history so far:")

messages = result["messages"]
count_tokens_approximately(messages)

"""
You can see that we are close to the `max_tokens` threshold, so on the next invocation we should see `pre_model_hook` kick-in and trim the message history. Let's run it again:
"""
logger.info("You can see that we are close to the `max_tokens` threshold, so on the next invocation we should see `pre_model_hook` kick-in and trim the message history. Let's run it again:")

inputs = {"messages": [("user", "where can i find the best bagel?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="updates"))

"""
You can see that the `pre_model_hook` node now only returned the last 3 messages, as expected. However, the existing message history is untouched:
"""
logger.info("You can see that the `pre_model_hook` node now only returned the last 3 messages, as expected. However, the existing message history is untouched:")

updated_messages = graph.get_state(config).values["messages"]
assert [(m.type, m.content) for m in updated_messages[: len(messages)]] == [
    (m.type, m.content) for m in messages
]

"""
## Overwrite the original message history

Let's now change the `pre_model_hook` to **overwrite** the message history in the graph state. To do this, we’ll return the updated messages under `messages` key. We’ll also include a special `RemoveMessage(REMOVE_ALL_MESSAGES)` object, which tells `create_react_agent` to remove previous messages from the graph state:
"""
logger.info("## Overwrite the original message history")


def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES)] + trimmed_messages}


checkpointer = InMemorySaver()
graph = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)

"""
Now let's run the agent with the same queries as before:
"""
logger.info("Now let's run the agent with the same queries as before:")

config = {"configurable": {"thread_id": "1"}}

inputs = {"messages": [("user", "What's the weather in NYC?")]}
result = graph.invoke(inputs, config=config)

inputs = {"messages": [("user", "What's it known for?")]}
result = graph.invoke(inputs, config=config)
messages = result["messages"]

inputs = {"messages": [("user", "where can i find the best bagel?")]}
print_stream(
    graph.stream(inputs, config=config, stream_mode="updates"),
    output_messages_key="messages",
)

"""
You can see that the `pre_model_hook` node returned the last 3 messages again. However, this time, the message history is modified in the graph state as well:
"""
logger.info("You can see that the `pre_model_hook` node returned the last 3 messages again. However, this time, the message history is modified in the graph state as well:")

updated_messages = graph.get_state(config).values["messages"]
assert (
    [(m.type, m.content) for m in updated_messages[:2]]
    == [(m.type, m.content) for m in messages[-2:]]
)

"""
## Summarizing message history

Finally, let's apply a different strategy for managing message history — summarization. Just as with trimming, you can choose to keep original message history unmodified or overwrite it. The example below will only show the former.

We will use the [`SummarizationNode`](https://langchain-ai.github.io/langmem/guides/summarization/#using-summarizationnode) from the prebuilt `langmem` library. Once the message history reaches the token limit, the summarization node will summarize earlier messages to make sure they fit into `max_tokens`.
"""
logger.info("## Summarizing message history")


model = ChatOllama(model="llama3.2")
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)


class State(AgentState):
    context: dict[str, Any]


checkpointer = InMemorySaver()
graph = create_react_agent(
    model.bind(max_tokens=256),
    tools,
    pre_model_hook=summarization_node,
    state_schema=State,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("user", "What's the weather in NYC?")]}

result = graph.invoke(inputs, config=config)

inputs = {"messages": [("user", "What's it known for?")]}
result = graph.invoke(inputs, config=config)

inputs = {"messages": [("user", "where can i find the best bagel?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="updates"))

"""
You can see that the earlier messages have now been replaced with the summary of the earlier conversation!
"""
logger.info("You can see that the earlier messages have now been replaced with the summary of the earlier conversation!")

logger.info("\n\n[DONE]", bright=True)
