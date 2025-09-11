from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.file.utils import save_file
from jet.logger import logger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
import json
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
# How to create a ReAct agent from scratch

!!! info "Prerequisites"
    This guide assumes familiarity with the following:
    
    - [Tool calling agent](../../concepts/agentic_concepts/#tool-calling-agent)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
    - [Messages](https://python.langchain.com/docs/concepts/messages/)
    - [LangGraph Glossary](../../concepts/low_level/)

Using the prebuilt ReAct agent [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent] is a great way to get started, but sometimes you might want more control and customization. In those cases, you can create a custom ReAct agent. This guide shows how to implement ReAct agent from scratch using LangGraph.

## Setup

First, let's install the required packages and set our API keys:
"""
logger.info("# How to create a ReAct agent from scratch")

# %%capture --no-stderr
# %pip install -U langgraph langchain-ollama

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

#         _set_env("OPENAI_API_KEY")


"""
<div class="admonition tip">
     <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for better debugging</p>
     <p style="padding-top: 5px;">
         Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM aps built with LangGraph — read more about how to get started in the <a href="https://docs.smith.langchain.com">docs</a>. 
     </p>
 </div>

## Create ReAct agent

Now that you have installed the required packages and set your environment variables, we can code our ReAct agent!

### Define graph state

We are going to define the most basic ReAct state in this example, which will just contain a list of messages.

For your specific use case, feel free to add any other state keys that you need.
"""
logger.info("## Create ReAct agent")


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


"""
### Define model and tools

Next, let's define the tools and model we will use for our example.
"""
logger.info("### Define model and tools")


model = ChatOllama(model="llama3.2")


@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"I am not sure what the weather is in {location}"


tools = [get_weather]

model = model.bind_tools(tools)

"""
### Define nodes and edges

Next let's define our nodes and edges. In our basic ReAct agent there are only two nodes, one for calling the model and one for using tools, however you can modify this basic structure to work better for your use case. The tool node we define here is a simplified version of the prebuilt [`ToolNode`](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/), which has some additional features.

Perhaps you want to add a node for [adding structured output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/) or a node for executing some external action (sending an email, adding a calendar event, etc.). Maybe you just want to change the way the `call_model` node works and how `should_continue` decides whether to call tools - the possibilities are endless and LangGraph makes it easy to customize this basic structure for your specific use case.
"""
logger.info("### Define nodes and edges")


tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(
            tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


"""
### Define the graph

Now that we have defined all of our nodes and edges, we can define and compile our graph. Depending on if you have added more nodes or different edges, you will need to edit this to fit your specific use case.
"""
logger.info("### Define the graph")


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass

render_mermaid_graph(graph, f"{OUTPUT_DIR}/graph_output.png")

"""
## Use ReAct agent

Now that we have created our react agent, let's actually put it to the test!
"""
logger.info("## Use ReAct agent")


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            logger.debug(message)
        else:
            logger.teal(message)


inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))

save_file(graph, f"{OUTPUT_DIR}/workflow_state.json")

"""
Perfect! The graph correctly calls the `get_weather` tool and responds to the user after receiving the information from the tool.
"""
logger.info("Perfect! The graph correctly calls the `get_weather` tool and responds to the user after receiving the information from the tool.")

logger.info("\n\n[DONE]", bright=True)
