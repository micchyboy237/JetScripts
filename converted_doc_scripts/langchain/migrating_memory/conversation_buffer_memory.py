from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from jet.llm.ollama.base_langchain import ChatOllama
import uuid
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from jet.llm.ollama.base_langchain import ChatOllama
import uuid
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from jet.llm.ollama.base_langchain import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# Migrating off ConversationBufferMemory or ConversationStringBufferMemory

[ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html)
and [ConversationStringBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationStringBufferMemory.html)
 were used to keep track of a conversation between a human and an ai asstistant without any additional processing. 


:::note
The `ConversationStringBufferMemory` is equivalent to `ConversationBufferMemory` but was targeting LLMs that were not chat models.
:::

The methods for handling conversation history using existing modern primitives are:

1. Using [LangGraph persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) along with appropriate processing of the message history
2. Using LCEL with [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#) combined with appropriate processing of the message history.

Most users will find [LangGraph persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) both easier to use and configure than the equivalent LCEL, especially for more complex use cases.

## Set up
"""

# %%capture --no-stderr
# %pip install --upgrade --quiet langchain-openai langchain

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
## Usage with LLMChain / ConversationChain

This section shows how to migrate off `ConversationBufferMemory` or `ConversationStringBufferMemory` that's used together with either an `LLMChain` or a `ConversationChain`.

### Legacy

Below is example usage of `ConversationBufferMemory` with an `LLMChain` or an equivalent `ConversationChain`.

<details open>
"""


# prompt = ChatPromptTemplate(
#     [
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{text}"),
#     ]
# )

# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True)

# legacy_chain = LLMChain(
#     llm=ChatOllama(model="llama3.1"),
#     prompt=prompt,
#     memory=memory,
# )

# legacy_result = legacy_chain.invoke({"text": "my name is bob"})
# logger.debug(legacy_result)

# legacy_result = legacy_chain.invoke({"text": "what was my name"})

# legacy_result["text"]

"""
:::note
Note that there is no support for separating conversation threads in a single memory object
:::

</details>

### LangGraph

The example below shows how to use LangGraph to implement a `ConversationChain` or `LLMChain` with `ConversationBufferMemory`.

This example assumes that you're already somewhat familiar with `LangGraph`. If you're not, then please see the [LangGraph Quickstart Guide](https://langchain-ai.github.io/langgraph/tutorials/introduction/) for more details.

`LangGraph` offers a lot of additional functionality (e.g., time-travel and interrupts) and will work well for other more complex (and realistic) architectures.

<details open>
"""


workflow = StateGraph(state_schema=MessagesState)

model = ChatOllama(model="llama3.1")


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
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
    logger.debug(event["messages"][-1].content)

input_message = HumanMessage(content="what was my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1].content)

"""
</details>

### LCEL RunnableWithMessageHistory

Alternatively, if you have a simple chain, you can wrap the chat model of the chain within a [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html).

Please refer to the following [migration guide](/docs/versions/migrating_chains/conversation_chain/) for more information.


## Usage with a pre-built agent

This example shows usage of an Agent Executor with a pre-built agent constructed using the [create_tool_calling_agent](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html) function.

If you are using one of the [old LangChain pre-built agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/), you should be able
to replace that code with the new [langgraph pre-built agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) which leverages
native tool calling capabilities of chat models and will likely work better out of the box.

### Legacy Usage

<details open>
"""


# model = ChatOllama(model="llama3.1")


# @tool
# def get_user_age(name: str) -> str:
#     """Use this tool to find the user's age."""
#     if "bob" in name.lower():
#         return "42 years old"
#     return "41 years old"


# tools = [get_user_age]

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

# agent = create_tool_calling_agent(model, tools, prompt)
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True)

# agent = create_tool_calling_agent(model, tools, prompt)
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     memory=memory,  # Pass the memory to the executor
# )

# logger.debug(agent_executor.invoke(
#     {"input": "hi! my name is bob what is my age?"}))
# logger.debug()
# logger.debug(agent_executor.invoke({"input": "do you remember my name?"}))

"""
</details>

### LangGraph

You can follow the standard LangChain tutorial for [building an agent](/docs/tutorials/agents/) an in depth explanation of how this works.

This example is shown here explicitly to make it easier for users to compare the legacy implementation vs. the corresponding langgraph implementation.

This example shows how to add memory to the [pre-built react agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) in langgraph.

For more details, please see the [how to add memory to the prebuilt ReAct agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-memory/) guide in langgraph.

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
app = create_react_agent(
    model,
    tools=[get_user_age],
    checkpointer=memory,
)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

input_message = HumanMessage(content="hi! I'm bob. What is my age?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1].content)

input_message = HumanMessage(content="do you remember my name?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1].content)

"""
If we use a different thread ID, it'll start a new conversation and the bot will not know our name!
"""

config = {"configurable": {"thread_id": "123456789"}}

input_message = HumanMessage(content="hi! do you remember my name?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    logger.debug(event["messages"][-1].content)

"""
</details>

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
