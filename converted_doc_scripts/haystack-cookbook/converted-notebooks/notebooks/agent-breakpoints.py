from haystack import Pipeline
from haystack.components.agents.agent import Agent
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool
from jet.logger import logger
from typing import Optional
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
# Breakpoints for Agent in a Pipeline

This notebook demonstrates how to set up breakpoints within an `Agent` component in a Haystack pipeline. Breakpoints can be placed either on the `chat_generator` or on any of the `tools` used by the `Agent`. This guide showcases both approaches.

The pipeline features an `Agent` acting as a database assistant, responsible for extracting relevant information and writing it to the database.

## Install packages
"""
logger.info("# Breakpoints for Agent in a Pipeline")

# %%bash

pip install "haystack-ai>=2.16.1"
pip install "transformers[torch,sentencepiece]"
pip install "sentence-transformers>=3.0.0"

"""
Setup Ollama API key for the `chat_generator`
"""
logger.info("Setup Ollama API key for the `chat_generator`")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter Ollama API key:")

"""
## Initializations

Now we initialize the components required to build an agentic pipeline. We will set up:

- A `chat_generator` for the Agent
- A custom `tool` that writes structured information to an `InMemoryDocumentStore`
- An `Agent` that uses the these components to extract and store entities from user-supplied context
"""
logger.info("## Initializations")



document_store = InMemoryDocumentStore()
chat_generator = OpenAIChatGenerator(
    model="llama3.2",
)

@tool
def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
    document_store.write_documents(
        [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other":other})]
    )

database_assistant = Agent(
        chat_generator=chat_generator,
        tools=[add_database_tool],
        system_prompt="""
        You are a database assistant.
        Your task is to extract the names of people mentioned in the given context and add them to a knowledge base,
        along with additional relevant information about them that can be extracted from the context.
        Do not use your own knowledge, stay grounded to the given context.
        Do not ask the user for confirmation. Instead, automatically update the knowledge base and return a brief
        summary of the people added, including the information stored for each.
        """,
        exit_conditions=["text"],
        max_agent_steps=100,
        raise_on_tool_invocation_failure=False
    )

"""
## Initialize the Pipeline
In this step, we construct a Haystack pipeline that performs the following tasks:

- Fetches HTML content from a specified URL.
- Converts the HTML into Haystack Document objects.
- Builds a `prompt` from the extracted content.
- Passes the prompt to the previously defined Agent, which processes the context and writes relevant information to a document store.
"""
logger.info("## Initialize the Pipeline")


pipeline_with_agent = Pipeline()
pipeline_with_agent.add_component("fetcher", LinkContentFetcher())
pipeline_with_agent.add_component("converter", HTMLToDocument())
pipeline_with_agent.add_component("builder", ChatPromptBuilder(
    template=[ChatMessage.from_user("""
    {% for doc in docs %}
    {{ doc.content|default|truncate(25000) }}
    {% endfor %}
    """)],
    required_variables=["docs"]
))
pipeline_with_agent.add_component("database_agent", database_assistant)

pipeline_with_agent.connect("fetcher.streams", "converter.sources")
pipeline_with_agent.connect("converter.documents", "builder.docs")
pipeline_with_agent.connect("builder", "database_agent")

"""
## Set up Breakpoints
With our pipeline in place, we can now configure a breakpoint on the Agent. This allows us to pause the pipeline execution at a specific step—in this case, during the Agent's operation—and save the intermediate pipeline snapshot to an external file for inspection or debugging.

We’ll first create a `Breakpoint` for the `chat_generator` and then wrap it using `AgentBreakpoint`, which explicitly targets the `Agent` component in the pipeline.

Set the `snapshot_file_path` to indicate where you want to save the file.
"""
logger.info("## Set up Breakpoints")


agent_generator_breakpoint = Breakpoint(component_name="chat_generator", visit_count=0, snapshot_file_path="snapshots/")
agent_breakpoint = AgentBreakpoint(break_point=agent_generator_breakpoint, agent_name='database_agent')
pipeline_with_agent.run(
    data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
    break_point=agent_breakpoint,
)

"""
This will generate a JSON file, named after the agent and component associated with the breakpoint, in the "snapshosts" directory containing a snapshot of the Pipeline where the Agent is running as well as a snapshot of the Agent state at the time of breakpoint.
"""
logger.info("This will generate a JSON file, named after the agent and component associated with the breakpoint, in the "snapshosts" directory containing a snapshot of the Pipeline where the Agent is running as well as a snapshot of the Agent state at the time of breakpoint.")

# !ls snapshots/database_agent_chat*

"""
We can also place a breakpoint on the `tool` used by the `Agent`. This allows us to interrupt the pipeline execution at the point where the `tool` is invoked by the `tool_invoker`.

To achieve this, we initialize a `ToolBreakpoint` with the name of the target tool, wrap it with an `AgentBreakpoint`, and then run the pipeline with the configured breakpoint.
"""
logger.info("We can also place a breakpoint on the `tool` used by the `Agent`. This allows us to interrupt the pipeline execution at the point where the `tool` is invoked by the `tool_invoker`.")

agent_tool_breakpoint = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="add_database_tool", snapshot_file_path="snapshots")
agent_breakpoint = AgentBreakpoint(break_point=agent_tool_breakpoint, agent_name = 'database_agent')

pipeline_with_agent.run(
    data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
    break_point=agent_breakpoint,
)

"""
Similarly this will also generate a JSON file in the "snapshosts" directory named after the agent's name and the the "tool_invoker" component which handled the tools used by the Agent.
"""
logger.info("Similarly this will also generate a JSON file in the "snapshosts" directory named after the agent's name and the the "tool_invoker" component which handled the tools used by the Agent.")

# !ls snapshots/database_agent_tool_invoker*

"""
## Resuming from a break point

For debugging purposes the snapshot files can be inspected and edited, and later injected into a pipeline and resume the execution from the point where the breakpoint was triggered.

Once a pipeline execution has been interrupted, we can resume the `pipeline_with_agent` from that saved state.

To do this:
- Use `load_state()` to load the saved pipeline state from disk. This function converts the stored JSON file back into a Python dictionary representing the intermediate state.
- Pass this state as an argument to the `Pipeline.run()` method.

The pipeline will resume execution from where it left off and continue until completion.
"""
logger.info("## Resuming from a break point")


snapshot = load_pipeline_snapshot("snapshots/database_agent_chat_generator_2025_07_26_12_22_11.json")

result = pipeline_with_agent.run(
    data={},
    pipeline_snapshot=snapshot
)

logger.debug(result['database_agent']['last_message'].text)

logger.info("\n\n[DONE]", bright=True)