from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from langgraph.types import interrupt, Command
from typing_extensions import Literal
import os
import random
import shutil
import uuid


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
# How to add multi-turn conversation in a multi-agent application (functional API)

!!! info "Prerequisites"
    This guide assumes familiarity with the following:

    - [Multi-agent systems](../../concepts/multi_agent)
    - [Human-in-the-loop](../../concepts/human_in_the_loop)
    - [Functional API](../../concepts/functional_api)
    - [Command](../../concepts/low_level/#command)
    - [LangGraph Glossary](../../concepts/low_level/)


In this how-to guide, we’ll build an application that allows an end-user to engage in a *multi-turn conversation* with one or more agents. We'll create a node that uses an [`interrupt`](../../reference/types/#langgraph.types.interrupt) to collect user input and routes back to the **active** agent.

The agents will be implemented as tasks in a workflow that executes agent steps and determines the next action:

1. **Wait for user input** to continue the conversation, or
2. **Route to another agent** (or back to itself, such as in a loop) via a [**handoff**](../../concepts/multi_agent/#handoffs).

```python


# Define a tool to signal intent to hand off to a different agent
# Note: this is not using Command(goto) syntax for navigating to different agents:
# `workflow()` below handles the handoffs explicitly
@tool(return_direct=True)
def transfer_to_hotel_advisor():
    """
logger.info("# How to add multi-turn conversation in a multi-agent application (functional API)")Ask hotel advisor agent for help."""
    return "Successfully transferred to hotel advisor"


# define an agent
travel_advisor_tools = [transfer_to_hotel_advisor, ...]
travel_advisor = create_react_agent(model, travel_advisor_tools)


# define a task that calls an agent
@task
def call_travel_advisor(messages):
    response = travel_advisor.invoke({"messages": messages})
    return response["messages"]


# define the multi-agent network workflow
@entrypoint(checkpointer)
def workflow(messages):
    call_active_agent = call_travel_advisor
    while True:
        agent_messages = call_active_agent(messages).result()
        ai_msg = get_last_ai_msg(agent_messages)
        if not ai_msg.tool_calls:
            user_input = interrupt(value="Ready for user input.")
            messages = messages + [{"role": "user", "content": user_input}]
            continue

        messages = messages + agent_messages
        call_active_agent = get_next_agent(messages)
    return entrypoint.final(value=agent_messages[-1], save=messages)
```

## Setup

First, let's install the required packages
"""
logger.info("# define an agent")


# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

In this example we will build a team of travel assistant agents that can communicate with each other.

We will create 2 agents:

* `travel_advisor`: can help with travel destination recommendations. Can ask `hotel_advisor` for help.
* `hotel_advisor`: can help with hotel recommendations. Can ask `travel_advisor` for help.

This is a fully-connected network - every agent can talk to any other agent.
"""
logger.info("Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https: // docs.smith.langchain.com">here</a>.")


@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])


@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)"
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]


@tool(return_direct=True)
def transfer_to_hotel_advisor():
    """Ask hotel advisor agent for help."""
    return "Successfully transferred to hotel advisor"


@tool(return_direct=True)
def transfer_to_travel_advisor():
    """Ask travel advisor agent for help."""
    return "Successfully transferred to travel advisor"


"""
!!! note "Transfer tools"

    You might have noticed that we're using `@tool(return_direct=True)` in the transfer tools. This is done so that individual agents (e.g., `travel_advisor`) can exit the ReAct loop early once these tools are called. This is the desired behavior, as we want to detect when the agent calls this tool and hand control off _immediately_ to a different agent. 
    
    **NOTE**: This is meant to work with the prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] -- if you are building a custom agent, make sure to manually add logic for handling early exit for tools that are marked with `return_direct`.

Let's now create our agents using the prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] and our multi-agent workflow. Note that will be calling [`interrupt`][langgraph.types.interrupt] every time after we get the final response from each of the agents.
"""
logger.info("You might have noticed that we're using `@tool(return_direct=True)` in the transfer tools. This is done so that individual agents (e.g., `travel_advisor`) can exit the ReAct loop early once these tools are called. This is the desired behavior, as we want to detect when the agent calls this tool and hand control off _immediately_ to a different agent.")


model = ChatOllama(model="llama3.2")

travel_advisor_tools = [
    get_travel_recommendations,
    transfer_to_hotel_advisor,
]
travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    prompt=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)


@task
def call_travel_advisor(messages):
    response = travel_advisor.invoke({"messages": messages})
    return response["messages"]


hotel_advisor_tools = [get_hotel_recommendations, transfer_to_travel_advisor]
hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
        "You MUST include human-readable response before transferring to another agent."
    ),
)


@task
def call_hotel_advisor(messages):
    response = hotel_advisor.invoke({"messages": messages})
    return response["messages"]


checkpointer = InMemorySaver()


def string_to_uuid(input_string):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, input_string))


@entrypoint(checkpointer=checkpointer)
def multi_turn_graph(messages, previous):
    previous = previous or []
    messages = add_messages(previous, messages)
    call_active_agent = call_travel_advisor
    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)
        ai_msg = next(m for m in reversed(agent_messages)
                      if isinstance(m, AIMessage))
        if not ai_msg.tool_calls:
            user_input = interrupt(value="Ready for user input.")
            human_message = {
                "role": "user",
                "content": user_input,
                "id": string_to_uuid(user_input),
            }
            messages = add_messages(messages, [human_message])
            continue

        tool_call = ai_msg.tool_calls[-1]
        if tool_call["name"] == "transfer_to_hotel_advisor":
            call_active_agent = call_hotel_advisor
        elif tool_call["name"] == "transfer_to_travel_advisor":
            call_active_agent = call_travel_advisor
        else:
            raise ValueError(
                f"Expected transfer tool, got '{tool_call['name']}'")

    return entrypoint.final(value=agent_messages[-1], save=messages)


"""
## Test multi-turn conversation

Let's test a multi turn conversation with this application.
"""
logger.info("## Test multi-turn conversation")

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = [
    {
        "role": "user",
        "content": "i wanna go somewhere warm in the caribbean",
        "id": str(uuid.uuid4()),
    },
    Command(
        resume="could you recommend a nice hotel in one of the areas and tell me which area it is."
    ),
    Command(
        resume="i like the first one. could you recommend something to do near the hotel?"
    ),
]

for idx, user_input in enumerate(inputs):
    logger.debug()
    logger.debug(f"--- Conversation Turn {idx + 1} ---")
    logger.debug()
    logger.debug(f"User: {user_input}")
    logger.debug()
    for update in multi_turn_graph.stream(
        user_input,
        config=thread_config,
        stream_mode="updates",
    ):
        for node_id, value in update.items():
            if isinstance(value, list) and value:
                last_message = value[-1]
                if isinstance(last_message, dict) or last_message.type != "ai":
                    continue
                logger.debug(f"{node_id}: {last_message.content}")

logger.info("\n\n[DONE]", bright=True)
