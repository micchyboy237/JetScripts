from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
import asyncio
import os
import shutil
import time


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
keywords: [create_react_agent, create_react_agent()]
---

# How to migrate from legacy LangChain agents to LangGraph

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Agents](/docs/concepts/agents)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Tool calling](/docs/how_to/tool_calling/)

:::

Here we focus on how to move from legacy LangChain agents to more flexible [LangGraph](https://langchain-ai.github.io/langgraph/) agents.
LangChain agents (the [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor) in particular) have multiple configuration parameters.
In this notebook we will show how those parameters map to the LangGraph react agent executor using the [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) prebuilt helper method.


:::note
In LangGraph, the graph replaces LangChain's agent executor. It manages the agent's cycles and tracks the scratchpad as messages within its state. The LangChain "agent" corresponds to the prompt and LLM you've provided.
:::


#### Prerequisites

This how-to guide uses Ollama as the LLM. Install the dependencies to run.
"""
logger.info("# How to migrate from legacy LangChain agents to LangGraph")

# %%capture --no-stderr
# %pip install -U langgraph langchain langchain-ollama

"""
Then, set your Ollama API key.
"""
logger.info("Then, set your Ollama API key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API key:\n")

"""
## Basic Usage

For basic creation and usage of a tool-calling ReAct-style agent, the functionality is the same. First, let's define a model and tool(s), then we'll use those to create an agent.
"""
logger.info("## Basic Usage")


model = ChatOllama(model="llama3.2")


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"

"""
For the LangChain [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), we define a prompt with a placeholder for the agent's scratchpad. The agent can be invoked as follows:
"""
logger.info("For the LangChain [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), we define a prompt with a placeholder for the agent's scratchpad. The agent can be invoked as follows:")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})

"""
LangGraph's [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) manages a state that is defined by a list of messages. It will continue to process the list until there are no tool calls in the agent's output. To kick it off, we input a list of messages. The output will contain the entire state of the graph-- in this case, the conversation history.
"""
logger.info("LangGraph's [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) manages a state that is defined by a list of messages. It will continue to process the list until there are no tool calls in the agent's output. To kick it off, we input a list of messages. The output will contain the entire state of the graph-- in this case, the conversation history.")


langgraph_agent_executor = create_react_agent(model, tools)


messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
{
    "input": query,
    "output": messages["messages"][-1].content,
}

message_history = messages["messages"]

new_query = "Pardon?"

messages = langgraph_agent_executor.invoke(
    {"messages": message_history + [("human", new_query)]}
)
{
    "input": new_query,
    "output": messages["messages"][-1].content,
}

"""
## Prompt Templates

With legacy LangChain agents you have to pass in a prompt template. You can use this to control the agent.

With LangGraph [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent), by default there is no prompt. You can achieve similar control over the agent in a few ways:

1. Pass in a system message as input
2. Initialize the agent with a system message
3. Initialize the agent with a function to transform messages in the graph state before passing to the model.
4. Initialize the agent with a [Runnable](/docs/concepts/lcel) to transform messages in the graph state before passing to the model. This includes passing prompt templates as well.

Let's take a look at all of these below. We will pass in custom instructions to get the agent to respond in Spanish.

First up, using `AgentExecutor`:
"""
logger.info("## Prompt Templates")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})

"""
Now, let's pass a custom system message to [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent).

LangGraph's prebuilt `create_react_agent` does not take a prompt template directly as a parameter, but instead takes a [`prompt`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) parameter. This modifies the graph state before the llm is called, and can be one of four values:

- A `SystemMessage`, which is added to the beginning of the list of messages.
- A `string`, which is converted to a `SystemMessage` and added to the beginning of the list of messages.
- A `Callable`, which should take in full graph state. The output is then passed to the language model.
- Or a [`Runnable`](/docs/concepts/lcel), which should take in full graph state. The output is then passed to the language model.

Here's how it looks in action:
"""
logger.info("Now, let's pass a custom system message to [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent).")


system_message = "You are a helpful assistant. Respond only in Spanish."

langgraph_agent_executor = create_react_agent(model, tools, prompt=system_message)


messages = langgraph_agent_executor.invoke({"messages": [("user", query)]})

"""
We can also pass in an arbitrary function or a runnable. This function/runnable should take in a graph state and output a list of messages.
We can do all types of arbitrary formatting of messages here. In this case, let's add a SystemMessage to the start of the list of messages and append another user message at the end.
"""
logger.info("We can also pass in an arbitrary function or a runnable. This function/runnable should take in a graph state and output a list of messages.")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("placeholder", "{messages}"),
        ("user", "Also say 'Pandamonium!' after the answer."),
    ]
)



langgraph_agent_executor = create_react_agent(model, tools, prompt=prompt)


messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
logger.debug(
    {
        "input": query,
        "output": messages["messages"][-1].content,
    }
)

"""
## Memory

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could add chat [Memory](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.memory) so it can engage in a multi-turn conversation.
"""
logger.info("## Memory")


model = ChatOllama(model="llama3.2")
memory = InMemoryChatMessageHistory(session_id="test-session")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}
logger.debug(
    agent_with_chat_history.invoke(
        {"input": "Hi, I'm polly! What's the output of magic_function of 3?"}, config
    )["output"]
)
logger.debug("---")
logger.debug(agent_with_chat_history.invoke({"input": "Remember my name?"}, config)["output"])
logger.debug("---")
logger.debug(
    agent_with_chat_history.invoke({"input": "what was that output again?"}, config)[
        "output"
    ]
)

"""
### In LangGraph

Memory is just [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/), aka [checkpointing](https://langchain-ai.github.io/langgraph/reference/checkpoints/).

Add a `checkpointer` to the agent and you get chat memory for free.
"""
logger.info("### In LangGraph")


system_message = "You are a helpful assistant."

memory = MemorySaver()
langgraph_agent_executor = create_react_agent(
    model, tools, prompt=system_message, checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}
logger.debug(
    langgraph_agent_executor.invoke(
        {
            "messages": [
                ("user", "Hi, I'm polly! What's the output of magic_function of 3?")
            ]
        },
        config,
    )["messages"][-1].content
)
logger.debug("---")
logger.debug(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "Remember my name?")]}, config
    )["messages"][-1].content
)
logger.debug("---")
logger.debug(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "what was that output again?")]}, config
    )["messages"][-1].content
)

"""
## Iterating through steps

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could iterate over the steps using the [stream](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) (or async `astream`) methods or the [iter](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter) method. LangGraph supports stepwise iteration using [stream](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream)
"""
logger.info("## Iterating through steps")


model = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for step in agent_executor.stream({"input": query}):
    logger.debug(step)

"""
### In LangGraph

In LangGraph, things are handled natively using [stream](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.stream) or the asynchronous `astream` method.
"""
logger.info("### In LangGraph")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)

langgraph_agent_executor = create_react_agent(model, tools, prompt=prompt)

for step in langgraph_agent_executor.stream(
    {"messages": [("human", query)]}, stream_mode="updates"
):
    logger.debug(step)

"""
## `return_intermediate_steps`

### In LangChain

Setting this parameter on AgentExecutor allows users to access intermediate_steps, which pairs agent actions (e.g., tool invocations) with their outcomes.
"""
logger.info("## `return_intermediate_steps`")

agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
result = agent_executor.invoke({"input": query})
logger.debug(result["intermediate_steps"])

"""
### In LangGraph

By default the [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) in LangGraph appends all messages to the central state. Therefore, it is easy to see any intermediate steps by just looking at the full state.
"""
logger.info("### In LangGraph")


langgraph_agent_executor = create_react_agent(model, tools=tools)

messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})

messages

"""
## `max_iterations`

### In LangChain

`AgentExecutor` implements a `max_iterations` parameter, allowing users to abort a run that exceeds a specified number of iterations.
"""
logger.info("## `max_iterations`")

@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    return "Sorry, there was an error. Please try again."


tools = [magic_function]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
)

agent_executor.invoke({"input": query})

"""
### In LangGraph

In LangGraph this is controlled via `recursion_limit` configuration parameter.

Note that in `AgentExecutor`, an "iteration" includes a full turn of tool invocation and execution. In LangGraph, each step contributes to the recursion limit, so we will need to multiply by two (and add one) to get equivalent results.

If the recursion limit is reached, LangGraph raises a specific exception type, that we can catch and manage similarly to AgentExecutor.
"""
logger.info("### In LangGraph")


RECURSION_LIMIT = 2 * 3 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        logger.debug(chunk["messages"][-1])
except GraphRecursionError:
    logger.debug({"input": query, "output": "Agent stopped due to max iterations."})

"""
## `max_execution_time`

### In LangChain

`AgentExecutor` implements a `max_execution_time` parameter, allowing users to abort a run that exceeds a total time limit.
"""
logger.info("## `max_execution_time`")



@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    time.sleep(2.5)
    return "Sorry, there was an error. Please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=2,
    verbose=True,
)

agent_executor.invoke({"input": query})

"""
### In LangGraph

With LangGraph's react agent, you can control timeouts on two levels. 

You can set a `step_timeout` to bound each **step**:
"""
logger.info("### In LangGraph")


langgraph_agent_executor = create_react_agent(model, tools=tools)
langgraph_agent_executor.step_timeout = 2

try:
    for chunk in langgraph_agent_executor.stream({"messages": [("human", query)]}):
        logger.debug(chunk)
        logger.debug("------")
except TimeoutError:
    logger.debug({"input": query, "output": "Agent stopped due to a step timeout."})

"""
The other way to set a single max timeout for an entire run is to directly use the python stdlib [asyncio](https://docs.python.org/3/library/asyncio.html) library.
"""
logger.info("The other way to set a single max timeout for an entire run is to directly use the python stdlib [asyncio](https://docs.python.org/3/library/asyncio.html) library.")



langgraph_agent_executor = create_react_agent(model, tools=tools)


async def stream(langgraph_agent_executor, inputs):
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]}
    ):
        logger.debug(chunk)
        logger.debug("------")


try:
    task = asyncio.create_task(
        stream(langgraph_agent_executor, {"messages": [("human", query)]})
    )
    await asyncio.wait_for(task, timeout=3)
except asyncio.TimeoutError:
    logger.debug("Task Cancelled.")

"""
## `early_stopping_method`

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could configure an [early_stopping_method](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.early_stopping_method) to either return a string saying "Agent stopped due to iteration limit or time limit." (`"force"`) or prompt the LLM a final time to respond (`"generate"`).
"""
logger.info("## `early_stopping_method`")


model = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return "Sorry there was an error, please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, early_stopping_method="force", max_iterations=1
)

result = agent_executor.invoke({"input": query})
logger.debug("Output with early_stopping_method='force':")
logger.debug(result["output"])

"""
### In LangGraph

In LangGraph, you can explicitly handle the response behavior outside the agent, since the full state can be accessed.
"""
logger.info("### In LangGraph")


RECURSION_LIMIT = 2 * 1 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        logger.debug(chunk["messages"][-1])
except GraphRecursionError:
    logger.debug({"input": query, "output": "Agent stopped due to max iterations."})

"""
## `trim_intermediate_steps`

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), you could trim the intermediate steps of long-running agents using [trim_intermediate_steps](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.trim_intermediate_steps), which is either an integer (indicating the agent should keep the last N steps) or a custom function.

For instance, we could trim the value so the agent only sees the most recent intermediate step.
"""
logger.info("## `trim_intermediate_steps`")


model = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    logger.debug(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)


def trim_steps(steps: list):
    return []


agent_executor = AgentExecutor(
    agent=agent, tools=tools, trim_intermediate_steps=trim_steps
)


query = "Call the magic function 4 times in sequence with the value 3. You cannot call it multiple times at once."

for step in agent_executor.stream({"input": query}):
    pass

"""
### In LangGraph

We can use the [`prompt`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) just as before when passing in [prompt templates](#prompt-templates).
"""
logger.info("### In LangGraph")


magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    logger.debug(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]


def _modify_state_messages(state: AgentState):
    return [("system", "You are a helpful assistant"), state["messages"][0]]


langgraph_agent_executor = create_react_agent(
    model, tools, prompt=_modify_state_messages
)

try:
    for step in langgraph_agent_executor.stream(
        {"messages": [("human", query)]}, stream_mode="updates"
    ):
        pass
except GraphRecursionError as e:
    logger.debug("Stopping agent prematurely due to triggering stop condition")

"""
## Next steps

You've now learned how to migrate your LangChain agent executors to LangGraph.

Next, check out other [LangGraph how-to guides](https://langchain-ai.github.io/langgraph/how-tos/).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)