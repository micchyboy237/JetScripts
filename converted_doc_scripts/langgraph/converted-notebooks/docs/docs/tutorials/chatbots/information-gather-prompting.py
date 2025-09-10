from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Annotated
from typing import List
from typing import Literal
from typing_extensions import TypedDict
import os
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
# Prompt Generation from User Requirements

In this example we will create a chat bot that helps a user generate a prompt.
It will first collect requirements from the user, and then will generate the prompt (and refine it based on user input).
These are split into two separate states, and the LLM decides when to transition between them.

A graphical representation of the system can be found below.

![prompt-generator.png](attachment:18f6888d-c412-4c53-ac3c-239fb90d2b6c.png)

## Setup

First, let's install our required packages and set our Ollama API key (the LLM we will use)
"""
logger.info("# Prompt Generation from User Requirements")

# %%capture --no-stderr
# % pip install -U langgraph jet.adapters.langchain.chat_ollama

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

#         _set_env("OPENAI_API_KEY")


"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Gather information

First, let's define the part of the graph that will gather user requirements. This will be an LLM call with a specific system message. It will have access to a tool that it can call when it is ready to generate the prompt.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>
"""
logger.info("## Gather information")


template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


llm = ChatOllama(model="llama3.2")
llm_with_tool = llm.bind_tools([PromptInstructions])


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


"""
## Generate Prompt

We now set up the state that will generate the prompt.
This will require a separate system message, as well as a function to filter out all message PRIOR to the tool invocation (as that is when the previous state decided it was time to generate the prompt
"""
logger.info("## Generate Prompt")


prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""


def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


"""
## Define the state logic

This is the logic for what state the chatbot is in.
If the last message is a tool call, then we are in the state where the "prompt creator" (`prompt`) should respond.
Otherwise, if the last message is not a HumanMessage, then we know the human should respond next and so we are in the `END` state.
If the last message is a HumanMessage, then if there was a tool call previously we are in the `prompt` state.
Otherwise, we are in the "info gathering" (`info`) state.
"""
logger.info("## Define the state logic")


def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


"""
## Create the graph

We can now the create the graph.
We will use a SqliteSaver to persist conversation history.
"""
logger.info("## Create the graph")


class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = InMemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }


workflow.add_conditional_edges(
    "info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)


render_mermaid_graph(graph, f"{OUTPUT_DIR}/graph_output.png")

"""
## Use the graph

We can now use the created chatbot.
"""
logger.info("## Use the graph")


# Real-world sample inputs: Simulate a conversation for generating a RAG prompt
sample_inputs = [
    "hi!",
    "I want a prompt for RAG application.",
    "Objective: Answer user query using provided context. Variables: context and query. No constraints beyond accuracy. Requirements: Structured JSON output."
]

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

for user_input in sample_inputs:
    logger.info(f"User: {user_input}")
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        logger.info(f"AI: {last_message.content}")

    if output and "prompt" in output:
        logger.success(output)
        logger.info("Prompt generation complete!")

# Additional real-world sample: Marketing email prompt
logger.info("\n--- New Thread: Marketing Email Prompt ---")
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
sample_inputs2 = [
    "Hello, I need a prompt for generating marketing emails.",
    "Objective: Create engaging promotional content. Variables: product_name, target_audience, key_features. Constraints: Avoid spam language. Requirements: Include call to action, be persuasive."
]

for user_input in sample_inputs2:
    logger.info(f"User: {user_input}")
    for output in graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        logger.success(f"AI: {last_message.content}")

logger.info("\n\n[DONE]", bright=True)
