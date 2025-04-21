from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

    
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
---
sidebar_position: 1
---

# How to add memory to chatbots

A key feature of chatbots is their ability to use the content of previous conversational turns as context. This state management can take several forms, including:

- Simply stuffing previous messages into a chat model prompt.
- The above, but trimming old messages to reduce the amount of distracting information the model has to deal with.
- More complex modifications like synthesizing summaries for long running conversations.

We'll go into more detail on a few techniques below!

:::note

This how-to guide previously built a chatbot using [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html). You can access this version of the guide in the [v0.2 docs](https://python.langchain.com/v0.2/docs/how_to/chatbots_memory/).

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into new LangChain applications.

If your code is already relying on `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do **not** need to make any changes. We do not plan on deprecating this functionality in the near future as it works for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.

Please see [How to migrate to LangGraph Memory](/docs/versions/migrating_memory/) for more details.
:::

## Setup

# You'll need to install a few packages, and have your Ollama API key set as an environment variable named `OPENAI_API_KEY`:
"""

# %pip install --upgrade --quiet langchain langchain-openai langgraph

# import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
Let's also set up a chat model that we'll use for the below examples.
"""


model = ChatOllama(model="llama3.1")

"""
## Message passing

The simplest form of memory is simply passing chat history messages into a chain. Here's an example:
"""


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

ai_msg = chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)
logger.debug(ai_msg.content)

"""
We can see that by passing the previous conversation into a chain, it can use it as context to answer questions. This is the basic concept underpinning chatbot memory - the rest of the guide will demonstrate convenient techniques for passing or reformatting messages.

## Automatic history management

The previous examples pass messages to the chain (and model) explicitly. This is a completely acceptable approach, but it does require external management of new messages. LangChain also provides a way to build applications that have memory using LangGraph's [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/). You can [enable persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) in LangGraph applications by providing a `checkpointer` when compiling the graph.
"""


workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""
We'll pass the latest input to the conversation here and let LangGraph keep track of the conversation history using the checkpointer:
"""

app.invoke(
    {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
    config={"configurable": {"thread_id": "1"}},
)

app.invoke(
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config={"configurable": {"thread_id": "1"}},
)

"""
## Modifying chat history

Modifying stored chat messages can help your chatbot handle a variety of situations. Here are some examples:

### Trimming messages

LLMs and chat models have limited context windows, and even if you're not directly hitting limits, you may want to limit the amount of distraction the model has to deal with. One solution is trim the history messages before passing them to the model. Let's use an example history with the `app` we declared above:
"""

demo_ephemeral_chat_history = [
    HumanMessage(content="Hey there! I'm Nemo."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks!"),
]

app.invoke(
    {
        "messages": demo_ephemeral_chat_history
        + [HumanMessage(content="What's my name?")]
    },
    config={"configurable": {"thread_id": "2"}},
)

"""
We can see the app remembers the preloaded name.

But let's say we have a very small context window, and we want to trim the number of messages passed to the model to only the 2 most recent ones. We can use the built in [trim_messages](/docs/how_to/trim_messages/) util to trim messages based on their token count before they reach our prompt. In this case we'll count each message as 1 "token" and keep only the last two messages:
"""


trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages
    response = model.invoke(messages)
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""
Let's call this new app and check the response
"""

app.invoke(
    {
        "messages": demo_ephemeral_chat_history
        + [HumanMessage(content="What is my name?")]
    },
    config={"configurable": {"thread_id": "3"}},
)

"""
We can see that `trim_messages` was called and only the two most recent messages will be passed to the model. In this case, this means that the model forgot the name we gave it.

Check out our [how to guide on trimming messages](/docs/how_to/trim_messages/) for more.

### Summary memory

We can use this same pattern in other ways too. For example, we could use an additional LLM call to generate a summary of the conversation before calling our app. Let's recreate our chat history:
"""

demo_ephemeral_chat_history = [
    HumanMessage(content="Hey there! I'm Nemo."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks!"),
]

"""
And now, let's update the model-calling function to distill previous interactions into a summary:
"""


workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    if len(message_history) >= 4:
        last_human_message = state["messages"][-1]
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        summary_message = model.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        human_message = HumanMessage(content=last_human_message.content)
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = model.invoke([system_message] + state["messages"])

    return {"messages": message_updates}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""
Let's see if it remembers the name we gave it:
"""

app.invoke(
    {
        "messages": demo_ephemeral_chat_history
        + [HumanMessage("What did I say my name was?")]
    },
    config={"configurable": {"thread_id": "4"}},
)

"""
Note that invoking the app again will keep accumulating the history until it reaches the specified number of messages (four in our case). At that point we will generate another summary generated from the initial summary plus new messages and so on.
"""

logger.info("\n\n[DONE]", bright=True)