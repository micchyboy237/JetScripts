from jet.transformers.formatters import format_json
from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple
from typing import Literal
from typing import Union
from typing_extensions import TypedDict
import operator
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
# Plan-and-Execute

This notebook shows how to create a "plan-and-execute" style agent. This is heavily inspired by the [Plan-and-Solve](https://arxiv.org/abs/2305.04091) paper as well as the [Baby-AGI](https://github.com/yoheinakajima/babyagi) project.

The core idea is to first come up with a multi-step plan, and then go through that plan one item at a time.
After accomplishing a particular task, you can then revisit the plan and modify as appropriate.


The general computational graph looks like the following:

![plan-and-execute diagram](attachment:86cf6404-3d9b-41cb-ab97-5e451f576620.png)


This compares to a typical [ReAct](https://arxiv.org/abs/2210.03629) style agent where you think one step at a time.
The advantages of this "plan-and-execute" style agent are:

1. Explicit long term planning (which even really strong LLMs can struggle with)
2. Ability to use smaller/weaker models for the execution step, only using larger/better models for the planning step


The following walkthrough demonstrates how to do so in LangGraph. The resulting agent will leave a trace like the following example: ([link](https://smith.langchain.com/public/d46e24d3-dda6-44d5-9550-b618fca4e0d4/r)).

## Setup

First, we need to install the packages required.
"""
logger.info("# Plan-and-Execute")

# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain-community langchain-ollama tavily-python

"""
Next, we need to set API keys for Ollama(the LLM we will use) and Tavily (the search tool we will use)
"""
logger.info(
    "Next, we need to set API keys for Ollama(the LLM we will use) and Tavily (the search tool we will use)")

# import getpass

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("OPENAI_API_KEY")
# _set_env("TAVILY_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Define Tools

We will first define the tools we want to use. For this simple example, we will use a built-in search tool via Tavily. However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/how_to/custom_tools) on how to do that.
"""
logger.info("## Define Tools")

tools = [TavilySearchResults(max_results=3)]

"""
## Define our Execution Agent

Now we will create the execution agent we want to use to execute tasks. 
Note that for this example, we will be using the same execution agent for each task, but this doesn't HAVE to be the case.
"""
logger.info("## Define our Execution Agent")

llm = ChatOllama(model="llama3.2")
prompt = "You are a helpful assistant."
agent_executor = create_react_agent(llm, tools, prompt=prompt)

agent_executor.invoke(
    {"messages": [("user", "who is the winnner of the us open")]})

"""
## Define the State

Let's now start by defining the state the track for this agent.

First, we will need to track the current plan. Let's represent that as a list of strings.

Next, we should track previously executed steps. Let's represent that as a list of tuples (these tuples will contain the step and then the result)

Finally, we need to have some state to represent the final response as well as the original input.
"""
logger.info("## Define the State")


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


"""
## Planning Step

Let's now think about creating the planning step. This will use function calling to create a plan.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>
"""
logger.info("## Planning Step")


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOllama(
    model="llama3.2", temperature=0
).with_structured_output(Plan)

planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)

"""
## Re-Plan Step

Now, let's create a step that re-does the plan based on the result of the previous step.
"""
logger.info("## Re-Plan Step")


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOllama(
    model="llama3.2", temperature=0
).with_structured_output(Act)

"""
## Create the Graph

We can now create the graph!
"""
logger.info("## Create the Graph")


def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]}
    )
    logger.success(format_json(agent_response))
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    logger.success(format_json(plan))
    return {"plan": plan.steps}


def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    logger.success(format_json(output))
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)

workflow.add_node("agent", execute_step)

workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

workflow.add_edge("planner", "agent")

workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

app = workflow.compile()
render_mermaid_graph(app, f"{OUTPUT_DIR}/graph_output.png", xray=True)

config = {"recursion_limit": 50}
inputs = {
    "input": "what is the hometown of the mens 2024 Australia open winner?"}
for event in app.stream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            logger.debug(v)

"""
## Conclusion

Congrats on making a plan-and-execute agent! One known limitations of the above design is that each task is still executed in sequence, meaning embarrassingly parallel operations all add to the total execution time. You could improve on this by having each task represented as a DAG (similar to LLMCompiler), rather than a regular list.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)
