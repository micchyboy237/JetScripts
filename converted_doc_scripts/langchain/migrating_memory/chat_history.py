from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
import uuid
from langchain_core.chat_history import InMemoryChatMessageHistory
import uuid
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessageChunk


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# How to use BaseChatMessageHistory with LangGraph

:::info Prerequisites

This guide assumes familiarity with the following concepts:
* [Chat History](/docs/concepts/chat_history)
* [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
* [LangGraph](https://langchain-ai.github.io/langgraph/concepts/high_level/)
* [Memory](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#memory)
:::

We recommend that new LangChain applications take advantage of the [built-in LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to implement memory.

In some situations, users may need to keep using an existing persistence solution for chat message history.

Here, we will show how to use [LangChain chat message histories](https://python.langchain.com/docs/integrations/memory/) (implementations of [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)) with LangGraph.

## Set up
"""

# %%capture --no-stderr
# %pip install --upgrade --quiet langchain-anthropic langgraph

# from getpass import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass()

"""
## ChatMessageHistory

A message history needs to be parameterized by a conversation ID or maybe by the 2-tuple of (user ID, conversation ID).

Many of the [LangChain chat message histories](https://python.langchain.com/docs/integrations/memory/) will have either a `session_id` or some `namespace` to allow keeping track of different conversations. Please refer to the specific implementations to check how it is parameterized.

The built-in `InMemoryChatMessageHistory` does not contains such a parameterization, so we'll create a dictionary to keep track of the message histories.
"""


chats_by_session_id = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history


"""
## Use with LangGraph

Next, we'll set up a basic chat bot using LangGraph. If you're not familiar with LangGraph, you should look at the following [Quick Start Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/).

We'll create a [LangGraph node](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes) for the chat model, and manually manage the conversation history, taking into account the conversation ID passed as part of the RunnableConfig.

The conversation ID can be passed as either part of the RunnableConfig (as we'll do here), or as part of the [graph state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).
"""


builder = StateGraph(state_schema=MessagesState)

model = ChatOllama(model="llama3.1")


def call_model(state: MessagesState, config: RunnableConfig) -> list[BaseMessage]:
    if "configurable" not in config or "session_id" not in config["configurable"]:
        raise ValueError(
            "Make sure that the config includes the following information: {'configurable': {'session_id': 'some_value'}}"
        )
    chat_history = get_chat_history(config["configurable"]["session_id"])
    messages = list(chat_history.messages) + state["messages"]
    ai_message = model.invoke(messages)
    chat_history.add_messages(state["messages"] + [ai_message])
    return {"messages": ai_message}


builder.add_edge(START, "model")
builder.add_node("model", call_model)

graph = builder.compile()

session_id = uuid.uuid4()
config = {"configurable": {"session_id": session_id}}

input_message = HumanMessage(content="hi! I'm bob")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

input_message = HumanMessage(content="what was my name?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

"""
:::tip

This also supports streaming LLM content token by token if using langgraph >= 0.2.28.
:::
"""


first = True

for msg, metadata in graph.stream(
    {"messages": input_message}, config, stream_mode="messages"
):
    if msg.content and not isinstance(msg, HumanMessage):
        logger.debug(msg.content, end="|", flush=True)

"""
## Using With RunnableWithMessageHistory

This how-to guide used the `messages` and `add_messages` interface of `BaseChatMessageHistory` directly. 

Alternatively, you can use [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html), as [LCEL](/docs/concepts/lcel/) can be used inside any [LangGraph node](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes).

To do that replace the following code:

```python
def call_model(state: MessagesState, config: RunnableConfig) -> list[BaseMessage]:
    # highlight-start
    # Make sure that config is populated with the session id
    if "configurable" not in config or "session_id" not in config["configurable"]:
        raise ValueError(
            "You make sure that the config includes the following information: {'configurable': {'session_id': 'some_value'}}"
        )
    # Fetch the history of messages and append to it any new messages.
    chat_history = get_chat_history(config["configurable"]["session_id"])
    messages = list(chat_history.messages) + state["messages"]
    ai_message = model.invoke(messages)
    # Finally, update the chat message history to include
    # the new input message from the user together with the
    # response from the model.
    chat_history.add_messages(state["messages"] + [ai_message])
    # hilight-end
    return {"messages": ai_message}
```

With the corresponding instance of `RunnableWithMessageHistory` defined in your current application.

```python
runnable = RunnableWithMessageHistory(...) # From existing code

def call_model(state: MessagesState, config: RunnableConfig) -> list[BaseMessage]:
    # RunnableWithMessageHistory takes care of reading the message history
    # and updating it with the new human message and ai response.
    ai_message = runnable.invoke(state['messages'], config)
    return {
        "messages": ai_message
    }
```
"""

logger.info("\n\n[DONE]", bright=True)
