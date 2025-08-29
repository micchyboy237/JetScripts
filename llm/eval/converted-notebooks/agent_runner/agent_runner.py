from llama_index.core.agent import AgentRunner, ReActAgentWorker, ReActAgent
from llama_index.agent.openai import OllamaAgentWorker, OllamaAgent
from llama_index.core.agent import AgentRunner
import nest_asyncio
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.llms import ChatMessage
from jet.llm.ollama.base import Ollama
from typing import Sequence, List
import json
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Step-wise, Controllable Agents
#
# This notebook shows you how to use our brand-new lower-level agent API, which supports a host of functionalities beyond simply executing a user query to help you create tasks, iterate through steps, and control the inputs for each step.
#
# High-Level Agent Architecture
#
# Our "agents" are composed of `AgentRunner` objects that interact with `AgentWorkers`. `AgentRunner`s are orchestrators that store state (including conversational memory), create and maintain tasks, run steps through each task, and offer the user-facing, high-level interface for users to interact with.
#
# `AgentWorker`s **control the step-wise execution of a Task**. Given an input step, an agent worker is responsible for generating the next step. They can be initialized with parameters and act upon state passed down from the Task/TaskStep objects, but do not inherently store state themselves. The outer `AgentRunner` is responsible for calling an `AgentWorker` and collecting/aggregating the results.
#
# If you are building your own agent, you will likely want to create your own `AgentWorker`. See below for an example!
#
# Notebook Walkthrough
#
# This notebook shows you how to run step-wise execution and full-execution with agents.
# - We show you how to do execution with OllamaAgent (function calling)
# - We show you how to do execution with ReActAgent

# %pip install llama-index-agent-openai
# %pip install llama-index-llms-ollama

# !pip install llama-index


nest_asyncio.apply()


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

tools = [multiply_tool, add_tool]

llm = Ollama(model="llama3.2")

# Test Ollama Agent
#
# There's two main ways to initialize the agent.
# - **Option 1**: Initialize `OllamaAgent`. This is a simple subclass of `AgentRunner` that bundles the `OllamaAgentWorker` under the hood.
# - **Option 2**: Initialize `AgentRunner` with `OllamaAgentWorker`. Here you import the modules and compose your own agent.
#
# **NOTE**: The old OllamaAgent can still be imported via `from llama_index.agent import OldOllamaAgent`.


agent = OllamaAgent.from_tools(tools, llm=llm, verbose=True)

# Test E2E Chat
#
# Here we re-demonstrate the end-to-end execution of a user task through the `chat()` function.
#
# This will iterate step-wise until the agent is done with the current task.

agent.chat("Hi")

response = agent.chat("What is (121 * 3) + 42?")

response

# Test Step-Wise Execution
#
# Now let's show the lower-level API in action. We do the same thing, but break this down into steps.

task = agent.create_task("What is (121 * 3) + 42?")

step_output = agent.run_step(task.task_id)

step_output

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

print(step_output.is_last)

response = agent.finalize_response(task.task_id)
print(str(response))

# Test ReAct Agent
#
# We do the same experiments, but with ReAct.

llm = Ollama(model="llama3.1")


agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

agent.chat("Hi")

response = agent.chat("What is (121 * 3) + 42?")

response

task = agent.create_task("What is (121 * 3) + 42?")

step_output = agent.run_step(task.task_id)

step_output.output

step_output = agent.run_step(task.task_id)

step_output.output

step_output = agent.run_step(task.task_id)

step_output.output

# List Out Tasks
#
# There are 3 tasks, corresponding to the three runs above.

tasks = agent.list_tasks()
print(len(tasks))

task_state = tasks[-1]
task_state.task.input

completed_steps = agent.get_completed_steps(task_state.task.task_id)

len(completed_steps)

completed_steps[0]

for idx in range(len(completed_steps)):
    print(f"Step {idx}")
    print(f"Response: {completed_steps[idx].output.response}")
    print(f"Sources: {completed_steps[idx].output.sources}")

logger.info("\n\n[DONE]", bright=True)
