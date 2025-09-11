from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
AIMessage,
HumanMessage,
SystemMessage,
ToolMessage,
trim_messages,
)
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List
import os
import shutil
import tiktoken


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
# How to trim messages

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Messages](/docs/concepts/messages)
- [Chat models](/docs/concepts/chat_models)
- [Chaining](/docs/how_to/sequence/)
- [Chat history](/docs/concepts/chat_history)

The methods in this guide also require `langchain-core>=0.2.9`.

:::

All models have finite context windows, meaning there's a limit to how many [tokens](/docs/concepts/tokens/) they can take as input. If you have very long messages or a chain/agent that accumulates a long message history, you'll need to manage the length of the messages you're passing in to the model.

[trim_messages](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) can be used to reduce the size of a chat history to a specified token count or specified message count.


If passing the trimmed chat history back into a chat model directly, the trimmed chat history should satisfy the following properties:

1. The resulting chat history should be **valid**. Usually this means that the following properties should be satisfied:
   - The chat history **starts** with either (1) a `HumanMessage` or (2) a [SystemMessage](/docs/concepts/messages/#systemmessage) followed by a `HumanMessage`.
   - The chat history **ends** with either a `HumanMessage` or a `ToolMessage`.
   - A `ToolMessage` can only appear after an `AIMessage` that involved a tool call.

   This can be achieved by setting `start_on="human"` and `ends_on=("human", "tool")`.
3. It includes recent messages and drops old messages in the chat history.
   This can be achieved by setting `strategy="last"`.
4. Usually, the new chat history should include the `SystemMessage` if it
   was present in the original chat history since the `SystemMessage` includes
   special instructions to the chat model. The `SystemMessage` is almost always
   the first message in the history if present. This can be achieved by setting
   `include_system=True`.

## Trimming based on token count

Here, we'll trim the chat history based on token count. The trimmed chat history will produce a **valid** chat history that includes the `SystemMessage`.

To keep the most recent messages, we set `strategy="last"`.  We'll also set `include_system=True` to include the `SystemMessage`, and `start_on="human"` to make sure the resulting chat history is valid. 

This is a good default configuration when using `trim_messages` based on token count. Remember to adjust `token_counter` and `max_tokens` for your use case. Keep in mind that new queries added to the chat history will be included in the token count unless you trim prior to adding the new query.

Notice that for our `token_counter` we can pass in a function (more on that below) or a language model (since language models have a message token counting method). It makes sense to pass in a model when you're trimming your messages to fit into the context window of that specific model:
"""
logger.info("# How to trim messages")

pip install -qU langchain-ollama


messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("what do you call a speechless parrot"),
]


trim_messages(
    messages,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=45,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
    allow_partial=False,
)

"""
## Trimming based on message count

Alternatively, we can trim the chat history based on **message count**, by setting `token_counter=len`. In this case, each message will count as a single token, and `max_tokens` will control
the maximum number of messages.

This is a good default configuration when using `trim_messages` based on message count. Remember to adjust `max_tokens` for your use case.
"""
logger.info("## Trimming based on message count")

trim_messages(
    messages,
    strategy="last",
    token_counter=len,
    max_tokens=5,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

"""
## Advanced Usage

You can use `trim_messages` as a building-block to create more complex processing logic.

If we want to allow splitting up the contents of a message we can specify `allow_partial=True`:
"""
logger.info("## Advanced Usage")

trim_messages(
    messages,
    max_tokens=56,
    strategy="last",
    token_counter=count_tokens_approximately,
    include_system=True,
    allow_partial=True,
)

"""
By default, the `SystemMessage` will not be included, so you can drop it by either setting `include_system=False` or by dropping the `include_system` argument.
"""
logger.info("By default, the `SystemMessage` will not be included, so you can drop it by either setting `include_system=False` or by dropping the `include_system` argument.")

trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=count_tokens_approximately,
)

"""
We can perform the flipped operation of getting the *first* `max_tokens` by specifying `strategy="first"`:
"""
logger.info("We can perform the flipped operation of getting the *first* `max_tokens` by specifying `strategy="first"`:")

trim_messages(
    messages,
    max_tokens=45,
    strategy="first",
    token_counter=count_tokens_approximately,
)

"""
## Using `ChatModel` as a token counter

You can pass a ChatModel as a token-counter. This will use `ChatModel.get_num_tokens_from_messages`. Let's demonstrate how to use it with Ollama:
"""
logger.info("## Using `ChatModel` as a token counter")


trim_messages(
    messages,
    max_tokens=45,
    strategy="first",
    token_counter=ChatOllama(model="llama3.2"),
)

"""
## Writing a custom token counter

We can write a custom token counter function that takes in a list of messages and returns an int.
"""
logger.info("## Writing a custom token counter")

pip install -qU tiktoken




def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Approximately reproduce https://github.com/ollama/ollama-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For simplicity only supports str Message.contents.
    """
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


trim_messages(
    messages,
    token_counter=tiktoken_counter,
    strategy="last",
    max_tokens=45,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

"""
## Chaining

`trim_messages` can be used imperatively (like above) or declaratively, making it easy to compose with other components in a chain
"""
logger.info("## Chaining")

llm = ChatOllama(model="llama3.2")

trimmer = trim_messages(
    token_counter=llm,
    strategy="last",
    max_tokens=45,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

chain = trimmer | llm
chain.invoke(messages)

"""
Looking at [the LangSmith trace](https://smith.langchain.com/public/65af12c4-c24d-4824-90f0-6547566e59bb/r) we can see that before the messages are passed to the model they are first trimmed.

Looking at just the trimmer, we can see that it's a Runnable object that can be invoked like all Runnables:
"""
logger.info("Looking at [the LangSmith trace](https://smith.langchain.com/public/65af12c4-c24d-4824-90f0-6547566e59bb/r) we can see that before the messages are passed to the model they are first trimmed.")

trimmer.invoke(messages)

"""
## Using with ChatMessageHistory

Trimming messages is especially useful when [working with chat histories](/docs/how_to/message_history/), which can get arbitrarily long:
"""
logger.info("## Using with ChatMessageHistory")


chat_history = InMemoryChatMessageHistory(messages=messages[:-1])


def dummy_get_session_history(session_id):
    if session_id != "1":
        return InMemoryChatMessageHistory()
    return chat_history


trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human",
)

chain = trimmer | llm
chain_with_history = RunnableWithMessageHistory(chain, dummy_get_session_history)
chain_with_history.invoke(
    [HumanMessage("what do you call a speechless parrot")],
    config={"configurable": {"session_id": "1"}},
)

"""
Looking at [the LangSmith trace](https://smith.langchain.com/public/17dd700b-9994-44ca-930c-116e00997315/r) we can see that we retrieve all of our messages but before the messages are passed to the model they are trimmed to be just the system message and last human message.

## API reference

For a complete description of all arguments head to the [API reference](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)