from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import ChatModelTabs from "@theme/ChatModelTabs";
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
---
keywords: [memory]
---

# How to add message history

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chaining runnables](/docs/how_to/sequence/)
- [Prompt templates](/docs/concepts/prompt_templates)
- [Chat Messages](/docs/concepts/messages)
- [LangGraph persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

:::

:::note

This guide previously covered the [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html) abstraction. You can access this version of the guide in the [v0.2 docs](https://python.langchain.com/v0.2/docs/how_to/message_history/).

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into new LangChain applications.

If your code is already relying on `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do **not** need to make any changes. We do not plan on deprecating this functionality in the near future as it works for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.

Please see [How to migrate to LangGraph Memory](/docs/versions/migrating_memory/) for more details.
:::

Passing conversation state into and out a chain is vital when building a chatbot. LangGraph implements a built-in persistence layer, allowing chain states to be automatically persisted in memory, or external backends such as SQLite, Postgres or Redis. Details can be found in the LangGraph [persistence documentation](https://langchain-ai.github.io/langgraph/how-tos/persistence/).

In this guide we demonstrate how to add persistence to arbitrary LangChain runnables by wrapping them in a minimal LangGraph application. This lets us persist the message history and other elements of the chain's state, simplifying the development of multi-turn applications. It also supports multiple threads, enabling a single application to interact separately with multiple users.

## Setup

Let's initialize a chat model:


<ChatModelTabs
  customVarName="llm"
/>
"""
logger.info("# How to add message history")


llm = ChatOllama(model="llama3.2")

"""
## Example: message inputs

Adding memory to a [chat model](/docs/concepts/chat_models) provides a simple example. Chat models accept a list of messages as input and output a message. LangGraph includes a built-in `MessagesState` that we can use for this purpose.

Below, we:
1. Define the graph state to be a list of messages;
2. Add a single node to the graph that calls a chat model;
3. Compile the graph with an in-memory checkpointer to store messages between runs.

:::info

The output of a LangGraph application is its [state](https://langchain-ai.github.io/langgraph/concepts/low_level/). This can be any Python type, but in this context it will typically be a `TypedDict` that matches the schema of your runnable.

:::
"""
logger.info("## Example: message inputs")


workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""
When we run the application, we pass in a configuration `dict` that specifies a `thread_id`. This ID is used to distinguish conversational threads (e.g., between different users).
"""
logger.info("When we run the application, we pass in a configuration `dict` that specifies a `thread_id`. This ID is used to distinguish conversational threads (e.g., between different users).")

config = {"configurable": {"thread_id": "abc123"}}

"""
We can then invoke the application:
"""
logger.info("We can then invoke the application:")

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_logger.debug()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_logger.debug()

"""
Note that states are separated for different threads. If we issue the same query to a thread with a new `thread_id`, the model indicates that it does not know the answer:
"""
logger.info("Note that states are separated for different threads. If we issue the same query to a thread with a new `thread_id`, the model indicates that it does not know the answer:")

query = "What's my name?"
config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_logger.debug()

"""
## Example: dictionary inputs

LangChain runnables often accept multiple inputs via separate keys in a single `dict` argument. A common example is a prompt template with multiple parameters.

Whereas before our runnable was a chat model, here we chain together a prompt template and chat model.
"""
logger.info("## Example: dictionary inputs")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

runnable = prompt | llm

"""
For this scenario, we define the graph state to include these parameters (in addition to the message history). We then define a single-node graph in the same way as before.

Note that in the below state:
- Updates to the `messages` list will append messages;
- Updates to the `language` string will overwrite the string.
"""
logger.info("For this scenario, we define the graph state to include these parameters (in addition to the message history). We then define a single-node graph in the same way as before.")




class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    response = runnable.invoke(state)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}

input_dict = {
    "messages": [HumanMessage("Hi, I'm Bob.")],
    "language": "Spanish",
}
output = app.invoke(input_dict, config)
output["messages"][-1].pretty_logger.debug()

"""
## Managing message history

The message history (and other elements of the application state) can be accessed via `.get_state`:
"""
logger.info("## Managing message history")

state = app.get_state(config).values

logger.debug(f"Language: {state['language']}")
for message in state["messages"]:
    message.pretty_logger.debug()

"""
We can also update the state via `.update_state`. For example, we can manually append a new message:
"""
logger.info("We can also update the state via `.update_state`. For example, we can manually append a new message:")


_ = app.update_state(config, {"messages": [HumanMessage("Test")]})

state = app.get_state(config).values

logger.debug(f"Language: {state['language']}")
for message in state["messages"]:
    message.pretty_logger.debug()

"""
For details on managing state, including deleting messages, see the LangGraph documentation:
- [How to delete messages](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/)
- [How to view and update past graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)
"""
logger.info("For details on managing state, including deleting messages, see the LangGraph documentation:")


logger.info("\n\n[DONE]", bright=True)