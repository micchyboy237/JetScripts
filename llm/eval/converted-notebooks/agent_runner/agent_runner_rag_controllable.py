from llama_index.agent.openai import OllamaAgentWorker
from llama_index.agent.openai import OllamaAgentWorker, OllamaAgent
from llama_index.core.agent import AgentRunner, ReActAgent
import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from jet.llm.ollama.base import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Controllable Agents for RAG
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_runner/agent_runner_rag_controllable.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# Adding agentic capabilities on top of your RAG pipeline can allow you to reason over much more complex questions.
#
# But a big pain point for agents is the **lack of steerability/transparency**. An agent may tackle a user query through chain-of-thought/planning, which requires repeated calls to an LLM. During this process it can be hard to inspect what's going on, or stop/correct execution in the middle.
#
# This notebook shows you how to use our brand-new lower-level agent API, which allows controllable step-wise execution, on top of a RAG pipeline.
#
# We showcase this over Wikipedia documents.

# %pip install llama-index-agent-openai
# %pip install llama-index-llms-ollama

# !pip install llama-index

# Setup Data
#
# Here we load a simple dataset of different cities from Wikipedia.


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

# Download Data

# !mkdir -p 'data/10q/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'

# Load data

march_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_march_2022.pdf"]
).load_data()
june_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_june_2022.pdf"]
).load_data()
sept_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_sept_2022.pdf"]
).load_data()

# Build indices/query engines/tools


def get_tool(name, full_name, documents=None):
    if not os.path.exists(f"./data/{name}"):
        vector_index = VectorStoreIndex.from_documents(documents)
        vector_index.storage_context.persist(persist_dir=f"./data/{name}")
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{name}"),
        )
    query_engine = vector_index.as_query_engine(similarity_top_k=3, llm=llm)
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=name,
            description=(
                "Provides information about Uber quarterly financials ending"
                f" {full_name}"
            ),
        ),
    )
    return query_engine_tool


march_tool = get_tool("march_2022", "March 2022", documents=march_2022)
june_tool = get_tool("june_2022", "June 2022", documents=june_2022)
sept_tool = get_tool("sept_2022", "September 2022", documents=sept_2022)

query_engine_tools = [march_tool, june_tool, sept_tool]

# Setup Agent
#
# In this section we define our tools and setup the agent.


agent_llm = Ollama(model="llama3.2", request_timeout=300.0,
                   context_window=4096)

agent = ReActAgent.from_tools(
    query_engine_tools, llm=agent_llm, verbose=True, max_iterations=20
)

# Run Some Queries
#
# We now demonstrate the capabilities of our step-wise agent framework.
#
# We show how it can handle complex queries, both e2e as well as step by step.
#
# We can then show how we can steer the outputs.

# Out of the box
#
# Calling `chat` will attempt to run the task end-to-end, and we notice that it only ends up calling one tool.

response = agent.chat("Analyze the changes in R&D expenditures and revenue")

print(str(response))

# Test Step-Wise Execution
#
# The end-to-end chat didn't work. Let's try to break it down step-by-step, and inject our own feedback if things are going wrong.

task = agent.create_task("Analyze the changes in R&D expenditures and revenue")

# This returns a `Task` object, which contains the `input`, additional state in `extra_state`, and other fields.
#
# Now let's try executing a single step of this task.

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

# We run into the **same issue**. The query finished even though we haven't analyzed the docs yet! Can we add a user input?

step_output = agent.run_step(task.task_id, input="What about June?")

print(step_output.is_last)

step_output = agent.run_step(task.task_id, input="What about September?")

step_output = agent.run_step(task.task_id)

# Since the steps look good, we are now ready to call `finalize_response`, get back our response.
#
# This will also commit the task execution to the `memory` object present in our `agent_runner`. We can inspect it.

response = agent.finalize_response(task.task_id)

print(str(response))

# Setup Human In the Loop Chat
#
# With these capabilities, it's easy to setup human-in-the-loop (or LLM-in-the-loop) feedback when interacting with an agent, especially for long-running tasks.
#
# We setup a double-loop: one for the task (the user "chatting" with an agent), and the other to control the intermediate executions.

agent_llm = Ollama(model="llama3.2", request_timeout=300.0,
                   context_window=4096)

agent = ReActAgent.from_tools(
    query_engine_tools, llm=agent_llm, verbose=True, max_iterations=20
)


def chat_repl(exit_when_done: bool = True):
    """Chat REPL.

    Args:
        exit_when_done(bool): if True, automatically exit when step is finished.
            Set to False if you want to keep going even if step is marked as finished by the agent.
            If False, you need to explicitly call "exit" to finalize a task execution.

    """
    task_message = None
    while task_message != "exit":
        task_message = input(">> Human: ")
        if task_message == "exit":
            break

        task = agent.create_task(task_message)

        response = None
        step_output = None
        message = None
        while message != "exit":
            if message is None or message == "":
                step_output = agent.run_step(task.task_id)
            else:
                step_output = agent.run_step(task.task_id, input=message)
            if exit_when_done and step_output.is_last:
                print(
                    ">> Task marked as finished by the agent, executing task execution."
                )
                break

            message = input(
                ">> Add feedback during step? (press enter/leave blank to continue, and type 'exit' to stop): "
            )
            if message == "exit":
                break

        if step_output is None:
            print(">> You haven't run the agent. Task is discarded.")
        elif not step_output.is_last:
            print(">> The agent hasn't finished yet. Task is discarded.")
        else:
            response = agent.finalize_response(task.task_id)
        print(f"Agent: {str(response)}")


chat_repl()

logger.info("\n\n[DONE]", bright=True)
