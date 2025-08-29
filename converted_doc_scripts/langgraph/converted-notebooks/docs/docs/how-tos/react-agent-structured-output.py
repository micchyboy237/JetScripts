from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Literal
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# How to force tool-calling agent to structure output

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#structured-output">
                    Structured Output
                </a>
            </li>            
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent">
                    Tool calling agent
                </a>
            </li>                
            <li>
                <a href="https://python.langchain.com/docs/concepts/#chat-models">
                    Chat Models
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#messages">
                    Messages
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/">
                    LangGraph Glossary
                </a>
            </li>
        </ul>
    </p>
</div> 

You might want your agent to return its output in a structured format. For example, if the output of the agent is used by some other downstream software, you may want the output to be in the same structured format every time the agent is invoked to ensure consistency.

This notebook will walk through two different options for forcing a tool calling agent to structure its output. We will be using a basic [ReAct agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) (a model node and a tool-calling node) together with a third node at the end that will format response for the user. Both of the options will use the same graph structure as shown in the diagram below, but will have different mechanisms under the hood.

![react_diagrams.png](attachment:59e8ed35-f2b4-421e-8d21-880e7ab31e5f.png)

**Option 1**

![option1.png](attachment:f717c664-605d-48d7-b534-deec99087214.png)

The first way you can force your tool calling agent to have structured output is to bind the output you would like as an additional tool for the `agent` node to use. In contrast to the basic ReAct agent, the `agent` node in this case is not selecting between `tools` and `END` but rather selecting between the specific tools it calls. The expected flow in this case is that the LLM in the `agent` node will first select the action tool, and after receiving the action tool output it will call the response tool, which will then route to the `respond` node which simply structures the arguments from the `agent` node tool call.

**Pros and Cons**

The benefit to this format is that you only need one LLM, and can save money and latency because of this. The downside to this option is that it isn't guaranteed that the single LLM will call the correct tool when you want it to. We can help the LLM by setting `tool_choice` to `any` when we use `bind_tools` which forces the LLM to select at least one tool at every turn, but this is far from a foolproof strategy. In addition, another downside is that the agent might call *multiple* tools, so we need to check for this explicitly in our routing function (or if we are using Ollama we can set `parallell_tool_calling=False` to ensure only one tool is called at a time).

**Option 2**

![option2.png](attachment:e9ef3df1-dbc0-4ff0-8040-0280372d67ac.png)

The second way you can force your tool calling agent to have structured output is to use a second LLM (in this case `model_with_structured_output`) to respond to the user. 

In this case, you will define a basic ReAct agent normally, but instead of having the `agent` node choose between the `tools` node and ending the conversation, the `agent` node will choose between the `tools` node and the `respond` node. The `respond` node will contain a second LLM that uses structured output, and once called will return directly to the user. You can think of this method as basic ReAct with one extra step before responding to the user. 

**Pros and Cons**

The benefit of this method is that it guarantees structured output (as long as `.with_structured_output` works as expected with the LLM). The downside to using this approach is that it requires making an additional LLM call before responding to the user, which can increase costs as well as latency. In addition, by not providing the `agent` node LLM with information about the desired output schema there is a risk that the `agent` LLM will fail to call the correct tools required to answer in the correct output schema.

Note that both of these options will follow the exact same graph structure (see the diagram above), in that they are both exact replicas of the basic ReAct architecture but with a `respond` node before the end.

## Setup

First, let's install the required packages and set our API keys
"""
logger.info("# How to force tool-calling agent to structure output")

# %%capture --no-stderr
# %pip install -U langgraph jet.llm.ollama.base_langchain

# import getpass


def _set_env(var: str):
    if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Define model, tools, and graph state

Now we can define how we want to structure our output, define our graph state, and also our tools and the models we are going to use.

To use structured output, we will use the `with_structured_output` method from LangChain, which you can read more about [here](https://python.langchain.com/docs/how_to/structured_output/).

We are going to use a single tool in this example for finding the weather, and will return a structured weather response to the user.
"""
logger.info("## Define model, tools, and graph state")



class WeatherResponse(BaseModel):
    """Respond to the user with this"""

    temperature: float = Field(description="The temperature in fahrenheit")
    wind_directon: str = Field(
        description="The direction of the wind in abbreviated form"
    )
    wind_speed: float = Field(description="The speed of the wind in km/h")


class AgentState(MessagesState):
    final_response: WeatherResponse


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It is cloudy in NYC, with 5 mph winds in the North-East direction and a temperature of 70 degrees"
    elif city == "sf":
        return "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOllama(model="llama3.2")

model_with_tools = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(WeatherResponse)

"""
## Option 1: Bind output as tool

Let's now examine how we would use the single LLM option.

### Define Graph

The graph definition is very similar to the one above, the only difference is we no longer call an LLM in the `response` node, and instead bind the `WeatherResponse` tool to our LLM that already contains the `get_weather` tool.
"""
logger.info("## Option 1: Bind output as tool")


tools = [get_weather, WeatherResponse]

model_with_response_tool = model.bind_tools(tools, tool_choice="any")


def call_model(state: AgentState):
    response = model_with_response_tool.invoke(state["messages"])
    return {"messages": [response]}


def respond(state: AgentState):
    weather_tool_call = state["messages"][-1].tool_calls[0]
    response = WeatherResponse(**weather_tool_call["args"])
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": weather_tool_call["id"],
    }
    return {"final_response": response, "messages": [tool_message]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"] == "WeatherResponse"
    ):
        return "respond"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()

"""
### Usage

Now we can run our graph to check that it worked as intended:
"""
logger.info("### Usage")

answer = graph.invoke(input={"messages": [("human", "what's the weather in SF?")]})[
    "final_response"
]

answer

"""
Again, the agent returned a `WeatherResponse` object as we expected.

## Option 2: 2 LLMs

Let's now dive into how we would use a second LLM to force structured output.

### Define Graph

We can now define our graph:
"""
logger.info("## Option 2: 2 LLMs")



def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def respond(state: AgentState):
    response = model_with_structured_output.invoke(
        [HumanMessage(content=state["messages"][-2].content)]
    )
    return {"final_response": response}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "respond"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()

"""
### Usage

We can now invoke our graph to verify that the output is being structured as desired:
"""
logger.info("### Usage")

answer = graph.invoke(input={"messages": [("human", "what's the weather in SF?")]})[
    "final_response"
]

answer

"""
As we can see, the agent returned a `WeatherResponse` object as we expected. If would now be easy to use this agent in a more complex software stack without having to worry about the output of the agent not matching the format expected from the next step in the stack.
"""
logger.info("As we can see, the agent returned a `WeatherResponse` object as we expected. If would now be easy to use this agent in a more complex software stack without having to worry about the output of the agent not matching the format expected from the next step in the stack.")

logger.info("\n\n[DONE]", bright=True)