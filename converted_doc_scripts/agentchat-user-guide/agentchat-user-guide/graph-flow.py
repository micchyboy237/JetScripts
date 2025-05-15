import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import (
DiGraphBuilder,
GraphFlow,
)
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# GraphFlow (Workflows)

In this section you'll learn how to create an _multi-agent workflow_ using {py:class}`~autogen_agentchat.teams.GraphFlow`, or simply "flow" for short.
It uses structured execution and precisely controls how agents interact to accomplish a task.

We'll first show you how to create and run a flow. We'll then explain how to observe and debug flow behavior, 
and discuss important operations for managing execution.

AutoGen AgentChat provides a team for directed graph execution:

- {py:class}`~autogen_agentchat.teams.GraphFlow`: A team that follows a {py:class}`~autogen_agentchat.teams.DiGraph`
to control the execution flow between agents. 
Supports sequential, parallel, conditional, and looping behaviors.

```{note}
**When should you use {py:class}`~autogen_agentchat.teams.GraphFlow`?**

Use Graph when you need strict control over the order in which agents act, or when different outcomes must lead to different next steps.
Start with a simple team such as {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` or {py:class}`~autogen_agentchat.teams.SelectorGroupChat`
if ad-hoc conversation flow is sufficient. 
Transition to a structured workflow when your task requires deterministic control,
conditional branching, or handling complex multi-step processes with cycles.
```

> **Warning:** {py:class}`~autogen_agentchat.teams.GraphFlow` is an **experimental feature**. 
Its API, behavior, and capabilities are **subject to change** in future releases.

## Creating and Running a Flow

{py:class}`~autogen_agentchat.teams.DiGraphBuilder` is a fluent utility that lets you easily construct execution graphs for workflows. It supports building:

- Sequential chains
- Parallel fan-outs
- Conditional branching
- Loops with safe exit conditions

Each node in the graph represents an agent, and edges define the allowed execution paths. Edges can optionally have conditions based on agent messages.

### Sequential Flow

We will begin by creating a simple workflow where a **writer** drafts a paragraph and a **reviewer** provides feedback. This graph terminates after the reviewer comments on the writer. 

Note, the flow automatically computes all the source and leaf nodes of the graph and the execution starts at all the source nodes in the graph and completes execution when no nodes are left to execute.
"""
logger.info("# GraphFlow (Workflows)")


client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

writer = AssistantAgent("writer", model_client=client, system_message="Draft a short paragraph on climate change.")

reviewer = AssistantAgent("reviewer", model_client=client, system_message="Review the draft and suggest improvements.")

builder = DiGraphBuilder()
builder.add_node(writer).add_node(reviewer)
builder.add_edge(writer, reviewer)

graph = builder.build()

flow = GraphFlow([writer, reviewer], graph=graph)

stream = flow.run_stream(task="Write a short paragraph about climate change.")
async for event in stream:  # type: ignore
    logger.debug(event)

"""
### Parallel Flow with Join

We now create a slightly more complex flow:

- A **writer** drafts a paragraph.
- Two **editors** independently edit for grammar and style (parallel fan-out).
- A **final reviewer** consolidates their edits (join).

Execution starts at the **writer**, fans out to **editor1** and **editor2** simultaneously, and then both feed into the **final reviewer**.
"""
logger.info("### Parallel Flow with Join")


client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

writer = AssistantAgent("writer", model_client=client, system_message="Draft a short paragraph on climate change.")

editor1 = AssistantAgent("editor1", model_client=client, system_message="Edit the paragraph for grammar.")

editor2 = AssistantAgent("editor2", model_client=client, system_message="Edit the paragraph for style.")

final_reviewer = AssistantAgent(
    "final_reviewer",
    model_client=client,
    system_message="Consolidate the grammar and style edits into a final version.",
)

builder = DiGraphBuilder()
builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

builder.add_edge(writer, editor1)
builder.add_edge(writer, editor2)

builder.add_edge(editor1, final_reviewer)
builder.add_edge(editor2, final_reviewer)

graph = builder.build()

flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)

async def run_async_code_11628adb():
    await Console(flow.run_stream(task="Write a short paragraph about climate change."))
    return 
 = asyncio.run(run_async_code_11628adb())
logger.success(format_json())

"""
## Message Filtering

### Execution Graph vs. Message Graph

In {py:class}`~autogen_agentchat.teams.GraphFlow`, the **execution graph** is defined using 
{py:class}`~autogen_agentchat.teams.DiGraph`, which controls the order in which agents execute.
However, the execution graph does not control what messages an agent receives from other agents.
By default, all messages are sent to all agents in the graph.

**Message filtering** is a separate feature that allows you to filter the messages
received by each agent and limiting their model context to only the relevant information.
The set of message filters defines the **message graph** in the flow.

Specifying the message graph can help with:
- Reduce hallucinations
- Control memory load
- Focus agents only on relevant information

You can use {py:class}`~autogen_agentchat.agents.MessageFilterAgent` together with {py:class}`~autogen_agentchat.agents.MessageFilterConfig` and {py:class}`~autogen_agentchat.agents.PerSourceFilter` to define these rules.
"""
logger.info("## Message Filtering")


client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

researcher = AssistantAgent(
    "researcher", model_client=client, system_message="Summarize key facts about climate change."
)
analyst = AssistantAgent("analyst", model_client=client, system_message="Review the summary and suggest improvements.")
presenter = AssistantAgent(
    "presenter", model_client=client, system_message="Prepare a presentation slide based on the final summary."
)

filtered_analyst = MessageFilterAgent(
    name="analyst",
    wrapped_agent=analyst,
    filter=MessageFilterConfig(per_source=[PerSourceFilter(source="researcher", position="last", count=1)]),
)

filtered_presenter = MessageFilterAgent(
    name="presenter",
    wrapped_agent=presenter,
    filter=MessageFilterConfig(per_source=[PerSourceFilter(source="analyst", position="last", count=1)]),
)

builder = DiGraphBuilder()
builder.add_node(researcher).add_node(filtered_analyst).add_node(filtered_presenter)
builder.add_edge(researcher, filtered_analyst).add_edge(filtered_analyst, filtered_presenter)

flow = GraphFlow(
    participants=builder.get_participants(),
    graph=builder.build(),
)

async def run_async_code_4d74d6c5():
    await Console(flow.run_stream(task="Summarize key facts about climate change."))
    return 
 = asyncio.run(run_async_code_4d74d6c5())
logger.success(format_json())

"""
## 🔁 Advanced Example: Conditional Loop + Filtered Summary

This example demonstrates:

- A loop between generator and reviewer (which exits when reviewer says "APPROVE")
- A summarizer agent that only sees the first user input and the last reviewer message
"""
logger.info("## 🔁 Advanced Example: Conditional Loop + Filtered Summary")


model_client = OllamaChatCompletionClient(model="llama3.1")

generator = AssistantAgent("generator", model_client=model_client, system_message="Generate a list of creative ideas.")
reviewer = AssistantAgent(
    "reviewer",
    model_client=model_client,
    system_message="Review ideas and say 'REVISE' and provide feedbacks, or 'APPROVE' for final approval.",
)
summarizer_core = AssistantAgent(
    "summary", model_client=model_client, system_message="Summarize the user request and the final feedback."
)

filtered_summarizer = MessageFilterAgent(
    name="summary",
    wrapped_agent=summarizer_core,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="user", position="first", count=1),
            PerSourceFilter(source="reviewer", position="last", count=1),
        ]
    ),
)

builder = DiGraphBuilder()
builder.add_node(generator).add_node(reviewer).add_node(filtered_summarizer)
builder.add_edge(generator, reviewer)
builder.add_edge(reviewer, generator, condition="REVISE")
builder.add_edge(reviewer, filtered_summarizer, condition="APPROVE")
builder.set_entry_point(generator)  # Set entry point to generator. Required if there are no source nodes.
graph = builder.build()

flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)

async def run_async_code_a9317e94():
    await Console(flow.run_stream(task="Brainstorm ways to reduce plastic waste."))
    return 
 = asyncio.run(run_async_code_a9317e94())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)