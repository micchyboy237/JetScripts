from llama_index.core.agent import QueryPipelineAgentWorker
from IPython.display import display, HTML
from pyvis.network import Network
from jet.llm.ollama.base import Ollama
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.agent.types import Task
from llama_index.core.llms import ChatResponse
from llama_index.core.agent.react.output_parser import ReActOutputParser
from typing import Set, Optional
from llama_index.core.tools import BaseTool
from llama_index.core.llms import ChatMessage
from llama_index.core.query_pipeline import InputComponent, Link
from llama_index.core.agent import ReActChatFormatter
from typing import Dict, Any, Optional, Tuple, List, cast
from llama_index.core.llms import MessageRole
from llama_index.core.query_pipeline import (
    AgentInputComponent,
    AgentFnComponent,
    CustomAgentComponent,
    QueryComponent,
    ToolRunnerComponent,
)
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
import llama_index.core
import phoenix as px
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from IPython.display import Markdown, display
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import sys
import logging
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Building a Multi-PDF Agent using Query Pipelines and HyDE
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_runner/agent_around_query_pipeline_with_HyDE_for_PDFs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# In this example, we show you how to build a multi-PDF agent that can reason across multiple tools, each one corresponding to a RAG pipeline with HyDE over a document.
#
# Author: https://github.com/DoganK01
#
# **Install Dependencies**

# %pip install llama-index-llms-ollama
# %pip install llama-index
# %pip install pyvis

# %pip install arize-phoenix[evals]

# %pip install llama-index-callbacks-arize-phoenix

# **Download Data and Do Imports**

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# **Setup Observability**

callback_manager = CallbackManager()
Settings.callback_manager = callback_manager


px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")

# os.environ["OPENAI_API_KEY"] = "sk-"

# **Setup Multi Doc HyDE Query Engine / Tool**

# We setup HyDE Query engines and their tools for our multi doc system.
#
# HyDE, short for Hypothetical Document Embeddings, is an innovative retrieval technique aimed at bolstering the efficiency of document retrieval processes. This method operates by crafting a hypothetical document tailored to an incoming query, which is subsequently embedded. The resulting embedding is leveraged to efficiently retrieve real documents exhibiting similarities to the hypothetical counterpart.

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    lyft_index.storage_context.persist(persist_dir="./storage/lyft")
    uber_index.storage_context.persist(persist_dir="./storage/uber")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


hyde = HyDEQueryTransform(include_original=True)
lyft_hyde_query_engine = TransformQueryEngine(lyft_engine, hyde)
uber_hyde_query_engine = TransformQueryEngine(uber_engine, hyde)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_hyde_query_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_hyde_query_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

# **Setup ReAct Agent Pipeline**
#
# What is ReAct Agent
#
# *  ReAct is a technique that enables LLMs to reason and perform task-specific actions. It combines chain-of-thought reasoning with action planning. It enables LLMs to create reasoning tracks and task-specific actions, strengthening the synergy between them using memory.
#
# * *The ReACT agent model refers to a framework that integrates the reasoning capabilities of LLMs with the ability to take actionable steps, creating a more sophisticated system that can understand and process information, evaluate situations, take appropriate actions, communicate responses, and track ongoing situations.*
#
# ![image.png](data:image/png;base64,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)
#
# **Reasoning Loop** : The reasoning loop allows data agents to select and interact with tools in response to an input task.
#
# **Memory**: LLMs, with access to memory, can store and retrieve data, ideal for apps tracking state or accessing multiple sources. Memory retains past interactions, enabling seamless reference to earlier conversation points. This integration involves allocating memory slots for relevant information and leveraging retrieval mechanisms during conversations. By recalling stored data, LLMs enhance contextual responses and integrate external sources, enriching user experiences.
#
# Steps of the ReAct agent we will create
#
# 1.   Takes in agent inputs
# 2.   Calls ReAct prompt using LLM to generate next action/tool (or returns a response).
# 3. If tool/action is selected, call tool pipeline to execute tool + collect response (In this case, our tools are HyDE Query Engine tools for both documents).
# 4. If response is generated, get response.
#
#
#
# * An `AgentInputComponent `that allows you to convert the agent inputs (Task, state dictionary) into a set of inputs for the query pipeline.
#
# * An `AgentFnComponent`: a general processor that allows you to take in the current Task, state, as well as any arbitrary inputs, and returns an output. In this cookbook we define a function component to format the ReAct prompt. However, you can put this anywhere
#
# Note that any function passed into `AgentFnComponent` and `AgentInputComponent` MUST include task and state as input variables, as these are inputs passed from the agent.
#
# Note that the output of an agentic query pipeline MUST be `Tuple[AgentChatResponse, bool]`.
#
#
# Task and State
#
# **Task**: It contains the information required to fulfill the query requested by the user. User input, memory, metadatas, global states over time.
#
#
# **State**: Some informations like memory

# **Agent Input Component**
#
# Generates inputs for the given task.


def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


agent_input_component = AgentInputComponent(fn=agent_input_fn)

# Define Agent Prompt
# Here we define the agent component that generates a ReAct prompt, and after the output is generated from the LLM, parses into a structured object.
#
# After the input is received, LLM is called with the ReAct agent prompt.
#
# `ReActChatFormatter` basically generates a fully formatted react prompt using ReAct Prompting (Chain-Of-Thought + Acting)
#  method


def react_prompt_fn(
    task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )


react_prompt_component = AgentFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": query_engine_tools}
)

# You can see the ReAct prompt here:
#
# https://github.com/run-llama/llama_index/blob/6cd92affa5835aa21f823ff985a81f006c496bbd/llama-index-core/llama_index/core/agent/react/prompts.py#L6

# **Define Agent Output Parser + Tool Pipeline**
# Once the LLM gives an output, we have a decision tree:
#
# 1. If an answer is given, then we’re done. Process the output
#
# 2. If an action is given, we need to execute the specified tool with the specified args, and then process the output.
#
# Tool calling can be done via the `ToolRunnerComponent` module. This is a simple wrapper module that takes in a list of tools, and can be “executed” with the specified tool name (every tool has a name) and tool action.
#
# We implement this overall module `OutputAgentComponent` that subclasses `CustomAgentComponent`.
#
# `perse_react_output_fn` function simply parses the ReAct prompt got from `react_prompt_fn` into the reasoning step.
#
# In this case, the ReAct Agent choose whatever go with a tool or done with tools and simply gets the output that will be fit in chat response for agents (`AgentChatResponse`).
#
# The `run_tool_fn` function simply runs a tool if it is selected.
#
# Finally, the incoming output is edited in accordance with the Agent output format by applying the `process_agent_response_fn` function.


def parse_react_output_fn(
    task: Task, state: Dict[str, Any], chat_response: ChatResponse
):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


parse_react_output = AgentFnComponent(fn=parse_react_output_fn)


def run_tool_fn(
    task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep
):
    """Run tool and process tool output."""
    tool_runner_component = ToolRunnerComponent(
        query_engine_tools, callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(
        observation=str(tool_output["output"])
    )
    state["current_reasoning"].append(observation_step)

    return {"response_str": observation_step.get_content(), "is_done": False}


run_tool = AgentFnComponent(fn=run_tool_fn)


def process_response_fn(
    task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    response_str = response_step.response
    state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
    state["memory"].put(
        ChatMessage(content=response_str, role=MessageRole.ASSISTANT)
    )

    return {"response_str": response_str, "is_done": True}


process_response = AgentFnComponent(fn=process_response_fn)


def process_agent_response_fn(
    task: Task, state: Dict[str, Any], response_dict: dict
):
    """Process agent response."""
    return (
        AgentChatResponse(response_dict["response_str"]),
        response_dict["is_done"],
    )


process_agent_response = AgentFnComponent(fn=process_agent_response_fn)

# **Stitch together Agent Query Pipeline**
# We can now stitch together the top-level agent pipeline: agent_input -> react_prompt -> llm -> react_output.
#
# The last component is the if-else component that calls sub-components.


qp = QP(verbose=True)


qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": Ollama(model="llama3.1", request_timeout=300.0, context_window=4096),
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
        "process_agent_response": process_agent_response,
    }
)

qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

qp.add_link("process_response", "process_agent_response")
qp.add_link("run_tool", "process_agent_response")

# **Visualize Query Pipeline**


net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.clean_dag)
print(net)

net.write_html("agent_dag.html")


with open("agent_dag.html", "r") as file:
    html_content = file.read()

display(HTML(html_content))

# **Setup Agent Worker around our Query Engines**


agent_worker = QueryPipelineAgentWorker(qp)
agent = agent_worker.as_agent(
    callback_manager=CallbackManager([]), verbose=True
)

# **Run the Agent**

task = agent.create_task(
    "What was Uber's Management's Report on Internal Control over Financial Reporting?"
)

step_output = agent.run_step(task.task_id)

print(step_output)

task = agent.create_task("What was Lyft's revenue growth in 2021?")

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

step_output.is_last

print(step_output)

response = agent.finalize_response(task.task_id)

print(str(response))

logger.info("\n\n[DONE]", bright=True)
