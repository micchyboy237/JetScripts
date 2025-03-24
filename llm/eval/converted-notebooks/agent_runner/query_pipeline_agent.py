from typing import Tuple
from llama_index.core import Response
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import StatefulFnComponent
from typing import Dict, Any
from typing import Dict, Tuple, Any
from llama_index.core.agent import FnAgentWorker
from pyvis.network import Network
from jet.llm.ollama.base import Ollama
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
    StatefulFnComponent,
    QueryComponent,
    ToolRunnerComponent,
)
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import NLSQLTableQueryEngine
import llama_index.core
import phoenix as px
from llama_index.core.query_pipeline import QueryPipeline
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
from llama_index.core import SQLDatabase
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Building an Agent around a Query Pipeline
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_runner/query_pipeline_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# In this cookbook we show you how to build an agent around a query pipeline.
#
# Agents offer the ability to do complex, sequential reasoning on top of any query DAG that you have setup. Conceptually this is also one of the ways you can add a "loop" to the graph.
#
# We show you two examples of agents you can implement:
# - a full ReAct agent that can do tool picking
# - a "simple" agent that adds a retry layer around a text-to-sql query engine.
#
# **NOTE:** Any Text-to-SQL application should be aware that executing
# arbitrary SQL queries can be a security risk. It is recommended to
# take precautions as needed, such as using restricted roles, read-only
# databases, sandboxing, etc.


engine = create_engine("sqlite:///chinook.db")
sql_database = SQLDatabase(engine)


# Setup
#
# Setup Data
#
# We use the chinook database as sample data. [Source](https://www.sqlitetutorial.net/sqlite-sample-database/).

# %pip install llama-index-llms-ollama

# !curl "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O ./chinook.zip
# !unzip ./chinook.zip

# Setup Observability

# !python -m pip install --upgrade \
# openinference-instrumentation-llama-index \
# opentelemetry-sdk \
# opentelemetry-exporter-otlp \
# "opentelemetry-proto>=1.12.0"


px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")

# Setup Text-to-SQL Query Engine / Tool
#
# Now we setup a simple text-to-SQL tool: given a query, translate text to SQL, execute against database, and get back a result.


sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    verbose=True,
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description=(
        "Useful for translating a natural language query into a SQL query"
    ),
)

# Setup ReAct Agent Pipeline
#
# We now setup a ReAct pipeline for a single step using our Query Pipeline syntax. This is a multi-part process that does the following:
# 1. Takes in agent inputs
# 2. Calls ReAct prompt using LLM to generate next action/tool (or returns a response).
# 3. If tool/action is selected, call tool pipeline to execute tool + collect response.
# 4. If response is generated, get response.
#
# Throughout this we'll build a **stateful** agent pipeline. It contains the following components:
# - A `FnAgentWorker` - this is the agent that runs a stateful function. Within this stateful function we'll run a stateful query pipeline.
# - A `StatefulFnComponent` - these are present in the query pipeline. They track global state over query pipeline executions. They track the agent `task` and `step_state` as special keys by default.


qp = QP(verbose=True)

# Define Agent Input Component
#
# Here we define the agent input component, called at the beginning of every agent step. Besides passing along the input, we also do initialization/state modification.


def agent_input_fn(state: Dict[str, Any]) -> str:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    task = state["task"]
    if len(state["current_reasoning"]) == 0:
        reasoning_step = ObservationReasoningStep(observation=task.input)
        state["current_reasoning"].append(reasoning_step)
    return task.input


agent_input_component = StatefulFnComponent(fn=agent_input_fn)

# Define Agent Prompt
#
# Here we define the agent component that generates a ReAct prompt, and after the output is generated from the LLM, parses into a structured object.


def react_prompt_fn(
    state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    task = state["task"]
    chat_formatter = ReActChatFormatter()
    cur_prompt = chat_formatter.format(
        tools,
        chat_history=task.memory.get(),
        current_reasoning=state["current_reasoning"],
    )
    return cur_prompt


react_prompt_component = StatefulFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": [sql_tool]}
)

# Define Agent Output Parser + Tool Pipeline
#
# Once the LLM gives an output, we have a decision tree:
# 1. If an answer is given, then we're done. Process the output
# 2. If an action is given, we need to execute the specified tool with the specified args, and then process the output.
#
# Tool calling can be done via the `ToolRunnerComponent` module. This is a simple wrapper module that takes in a list of tools, and can be "executed" with the specified tool name (every tool has a name) and tool action.
#
# We implement this overall module `OutputAgentComponent` that subclasses `CustomAgentComponent`.
#
# Note: we also implement `sub_query_components` to pass through higher-level callback managers to the tool runner submodule.


def parse_react_output_fn(state: Dict[str, Any], chat_response: ChatResponse):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


parse_react_output = StatefulFnComponent(fn=parse_react_output_fn)


def run_tool_fn(state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    """Run tool and process tool output."""
    task = state["task"]
    tool_runner_component = ToolRunnerComponent(
        [sql_tool], callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(observation=str(tool_output))
    state["current_reasoning"].append(observation_step)

    return observation_step.get_content(), False


run_tool = StatefulFnComponent(fn=run_tool_fn)


def process_response_fn(
    state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    return response_step.response, True


process_response = StatefulFnComponent(fn=process_response_fn)

# Stitch together Agent Query Pipeline
#
# We can now stitch together the top-level agent pipeline: agent_input -> react_prompt -> llm -> react_output.
#
# The last component is the if-else component that calls sub-components.


qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": Ollama(model="llama3.1", request_timeout=300.0, context_window=4096),
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
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

# Visualize Query Pipeline


net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.clean_dag)
net.show("agent_dag.html")

# Setup Agent Worker around Text-to-SQL Query Pipeline
#
# Now that you've setup a query pipeline that can run the ReAct loop, let's put it inside a custom agent!
#
# Our custom agent implementation is implemented using a simple Python function plugged into a `FnAgentWorker`. This Python function will seed the query pipeline with the right state at a given step, and run it.
#
# Once a task is done, the agent also commits the input/response to its memory in the task module.


def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    if state["is_first"]:
        qp.set_state(
            {
                "task": task,
                "current_reasoning": [],
            }
        )
        state["is_first"] = False

    response_str, is_done = qp.run()
    state["__output__"] = response_str
    if is_done:
        task.memory.put_messages(
            [
                ChatMessage(content=task.input, role=MessageRole.USER),
                ChatMessage(content=response_str, role=MessageRole.ASSISTANT),
            ]
        )
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()

# Run the Agent
#
# Let's try the agent on some sample queries.

task = agent.create_task(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

step_output.is_last

response = agent.finalize_response(task.task_id)

print(str(response))

agent.reset()
response = agent.chat(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

print(str(response))

# Setup Simple Retry Agent Pipeline for Text-to-SQL
#
# Instead of the full ReAct pipeline that does tool picking, let's try a much simpler agent pipeline that only does text-to-SQL, with retry-logic.
#
# We try a simple text-based "retry" prompt where given the user input and previous conversation history, can generate a modified query that outputs the right result.

# Define Core Modules
#
# - agent input
# - retry prompt
# - output processor (including a validation prompt)


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)


def agent_input_fn(state: Dict[str, Any]) -> Dict:
    """Agent input function."""
    task = state["task"]
    state["convo_history"].append(f"User: {task.input}")
    convo_history_str = "\n".join(state["convo_history"]) or "None"
    return {"input": task.input, "convo_history": convo_history_str}


agent_input_component = StatefulFnComponent(fn=agent_input_fn)


retry_prompt_str = """\
You are trying to generate a proper natural language query given a user input.

This query will then be interpreted by a downstream text-to-SQL agent which
will convert the query to a SQL statement. If the agent triggers an error,
then that will be reflected in the current conversation history (see below).

If the conversation history is None, use the user input. If its not None,
generate a new SQL query that avoids the problems of the previous SQL query.

Input: {input}
Convo history (failed attempts): 
{convo_history}

New input: """
retry_prompt = PromptTemplate(retry_prompt_str)


validate_prompt_str = """\
Given the user query, validate whether the inferred SQL query and response from executing the query is correct and answers the query.

Answer with YES or NO.

Query: {input}
Inferred SQL query: {sql_query}
SQL Response: {sql_response}

Result: """
validate_prompt = PromptTemplate(validate_prompt_str)

MAX_ITER = 3


def agent_output_fn(
    state: Dict[str, Any], output: Response
) -> Tuple[AgentChatResponse, bool]:
    """Agent output component."""
    task = state["task"]
    print(f"> Inferred SQL Query: {output.metadata['sql_query']}")
    print(f"> SQL Response: {str(output)}")
    state["convo_history"].append(
        f"Assistant (inferred SQL query): {output.metadata['sql_query']}"
    )
    state["convo_history"].append(f"Assistant (response): {str(output)}")

    validate_prompt_partial = validate_prompt.as_query_component(
        partial={
            "sql_query": output.metadata["sql_query"],
            "sql_response": str(output),
        }
    )
    qp = QP(chain=[validate_prompt_partial, llm])
    validate_output = qp.run(input=task.input)

    state["count"] += 1
    is_done = False
    if state["count"] >= MAX_ITER:
        is_done = True
    if "YES" in validate_output.message.content:
        is_done = True

    return str(output), is_done


agent_output_component = StatefulFnComponent(fn=agent_output_fn)


qp = QP(
    modules={
        "input": agent_input_component,
        "retry_prompt": retry_prompt,
        "llm": llm,
        "sql_query_engine": sql_query_engine,
        "output_component": agent_output_component,
    },
    verbose=True,
)
qp.add_link(
    "input", "retry_prompt", dest_key="input", input_fn=lambda x: x["input"]
)
qp.add_link(
    "input",
    "retry_prompt",
    dest_key="convo_history",
    input_fn=lambda x: x["convo_history"],
)
qp.add_chain(["retry_prompt", "llm", "sql_query_engine", "output_component"])

# Visualize Query Pipeline


# Define Agent Worker


def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    if state["is_first"]:
        qp.set_state({"task": task, "convo_history": [], "count": 0})
        state["is_first"] = False

    response_str, is_done = qp.run()
    if is_done:
        state["__output__"] = response_str
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()

response = agent.chat(
    "How many albums did the artist who wrote 'Restless and Wild' release? (answer should be non-zero)?"
)
print(str(response))

logger.info("\n\n[DONE]", bright=True)
