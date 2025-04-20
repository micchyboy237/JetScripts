from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import trim_messages
from langchain_core.messages import trim_messages
import uuid
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from jet.llm.ollama.base_langchain import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from jet.llm.ollama.base_langchain import ChatOllama


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# Migrating off ConversationBufferWindowMemory or ConversationTokenBufferMemory

Follow this guide if you're trying to migrate off one of the old memory classes listed below:


| Memory Type                      | Description                                                                                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ConversationBufferWindowMemory` | Keeps the last `n` messages of the conversation. Drops the oldest messages when there are more than `n` messages.                                                                      |
| `ConversationTokenBufferMemory`  | Keeps only the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit. |

`ConversationBufferWindowMemory` and `ConversationTokenBufferMemory` apply additional processing on top of the raw conversation history to trim the conversation history to a size that fits inside the context window of a chat model. 

This processing functionality can be accomplished using LangChain's built-in [trim_messages](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) function.

:::important

We’ll begin by exploring a straightforward method that involves applying processing logic to the entire conversation history.

While this approach is easy to implement, it has a downside: as the conversation grows, so does the latency, since the logic is re-applied to all previous exchanges in the conversation at each turn.

More advanced strategies focus on incrementally updating the conversation history to avoid redundant processing.

For instance, the langgraph [how-to guide on summarization](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) demonstrates
how to maintain a running summary of the conversation while discarding older messages, ensuring they aren't re-processed during later turns.
:::

## Set up
"""

# %%capture --no-stderr
# %pip install --upgrade --quiet langchain-openai langchain

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
## Legacy usage with LLMChain / Conversation Chain

<details open>
"""


# prompt = ChatPromptTemplate(
#     [
#         SystemMessage(content="You are a helpful assistant."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{text}"),
#     ]
# )

# memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

# legacy_chain = LLMChain(
#     llm=ChatOllama(model="llama3.1"),
#     prompt=prompt,
#     memory=memory,
# )

# legacy_result = legacy_chain.invoke({"text": "my name is bob"})
# logger.debug(legacy_result)

# legacy_result = legacy_chain.invoke({"text": "what was my name"})
# logger.debug(legacy_result)

"""
</details>

## Reimplementing ConversationBufferWindowMemory logic

Let's first create appropriate logic to process the conversation history, and then we'll see how to integrate it into an application. You can later replace this basic setup with more advanced logic tailored to your specific needs.

We'll use `trim_messages` to implement logic that keeps the last `n` messages of the conversation. It will drop the oldest messages when the number of messages exceeds `n`.

In addition, we will also keep the system message if it's present -- when present, it's the first message in a conversation that includes instructions for the chat model.
"""


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
    HumanMessage("why is 42 always the answer?"),
    AIMessage(
        "Because it’s the only number that’s constantly right, even when it doesn’t add up!"
    ),
    HumanMessage("What did the cow say?"),
]


selected_messages = trim_messages(
    messages,
    token_counter=len,  # <-- len will simply count the number of messages rather than tokens
    max_tokens=5,  # <-- allow up to 5 messages.
    strategy="last",
    start_on="human",
    include_system=True,
    allow_partial=False,
)

for msg in selected_messages:
    logger.debug(msg)

"""
## Reimplementing ConversationTokenBufferMemory logic

Here, we'll use `trim_messages` to keeps the system message and the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit.
"""


selected_messages = trim_messages(
    messages,
    token_counter=ChatOllama(model="llama3.1"),
    max_tokens=80,  # <-- token limit
    start_on="human",
    include_system=True,
    strategy="last",
)

for msg in selected_messages:
    logger.debug(msg)

"""
## Modern usage with LangGraph

The example below shows how to use LangGraph to add simple conversation pre-processing logic.

:::note

If you want to avoid running the computation on the entire conversation history each time, you can follow
the [how-to guide on summarization](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) that demonstrates
how to discard older messages, ensuring they aren't re-processed during later turns.

:::

<details open>
"""


workflow = StateGraph(state_schema=MessagesState)

model = ChatOllama(model="llama3.1")


def call_model(state: MessagesState):
    selected_messages = trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    response = model.invoke(selected_messages)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory
)


thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

config = {"configurable": {"thread_id": thread_id}}
input_message = HumanMessage(content="what was my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

"""
</details>

## Usage with a pre-built langgraph agent

This example shows usage of an Agent Executor with a pre-built agent constructed using the [create_tool_calling_agent](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html) function.

If you are using one of the [old LangChain pre-built agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/), you should be able
to replace that code with the new [langgraph pre-built agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) which leverages
native tool calling capabilities of chat models and will likely work better out of the box.

<details open>
"""


@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"


memory = MemorySaver()
model = ChatOllama(model="llama3.1")


def prompt(state) -> list[BaseMessage]:
    """Given the agent state, return a list of messages for the chat model."""
    return trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


app = create_react_agent(
    model,
    tools=[get_user_age],
    checkpointer=memory,
    prompt=prompt,
)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

input_message = HumanMessage(content="hi! I'm bob. What is my age?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

input_message = HumanMessage(content="do you remember my name?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1])

"""
</details>

## LCEL: Add a preprocessing step

The simplest way to add complex conversation management is by introducing a pre-processing step in front of the chat model and pass the full conversation history to the pre-processing step.

This approach is conceptually simple and will work in many situations; for example, if using a [RunnableWithMessageHistory](/docs/how_to/message_history/) instead of wrapping the chat model, wrap the chat model with the pre-processor.

The obvious downside of this approach is that latency starts to increase as the conversation history grows because of two reasons:

1. As the conversation gets longer, more data may need to be fetched from whatever store your'e using to store the conversation history (if not storing it in memory).
2. The pre-processing logic will end up doing a lot of redundant computation, repeating computation from previous steps of the conversation.

:::caution

If you want to use a chat model's tool calling capabilities, remember to bind the tools to the model before adding the history pre-processing step to it!

:::

<details open>
"""


model = ChatOllama(model="llama3.1")


@tool
def what_did_the_cow_say() -> str:
    """Check to see what the cow said."""
    return "foo"


message_processor = trim_messages(  # Returns a Runnable if no messages are provided
    token_counter=len,  # <-- len will simply count the number of messages rather than tokens
    max_tokens=5,  # <-- allow up to 5 messages.
    strategy="last",
    start_on=("human", "ai"),
    include_system=True,  # <-- Keep the system message
    allow_partial=False,
)

model_with_tools = model.bind_tools([what_did_the_cow_say])

model_with_preprocessor = message_processor | model_with_tools

full_history = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("why is 42 always the answer?"),
    AIMessage(
        "Because it’s the only number that’s constantly right, even when it doesn’t add up!"
    ),
    HumanMessage("What did the cow say?"),
]


logger.debug(model_with_preprocessor.invoke(full_history))

"""
</details>

If you need to implement more efficient logic and want to use `RunnableWithMessageHistory` for now the way to achieve this
is to subclass from [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html) and
define appropriate logic for `add_messages` (that doesn't simply append the history, but instead re-writes it).

Unless you have a good reason to implement this solution, you should instead use LangGraph.

## Next steps

Explore persistence with LangGraph:

* [LangGraph quickstart tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
* [How to add persistence ("memory") to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
* [How to manage conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
* [How to add summary of the conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/)

Add persistence with simple LCEL (favor langgraph for more complex use cases):

* [How to add message history](/docs/how_to/message_history/)

Working with message history:

* [How to trim messages](/docs/how_to/trim_messages)
* [How to filter messages](/docs/how_to/filter_messages/)
* [How to merge message runs](/docs/how_to/merge_message_runs/)
"""


logger.info("\n\n[DONE]", bright=True)
