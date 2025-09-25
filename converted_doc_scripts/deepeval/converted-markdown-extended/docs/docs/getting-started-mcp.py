from jet.transformers.formatters import format_json
from contextlib import AsyncExitStack
from deepeval import evaluate
from deepeval.metrics import MCPUseMetric
from deepeval.metrics import MultiTurnMCPUseMetric, MCPTaskCompletionMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import LLMTestCase
from deepeval.test_case import MCPServer
from deepeval.test_case import MCPToolCall
from deepeval.test_case import MCPToolCall, Turn
from jet.logger import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import mcp
import os
import shutil
import { Timeline, TimelineItem } from "@site/src/components/Timeline";


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
---
id: getting-started-mcp
title: MCP Evaluation
sidebar_label: MCP
---


Learn to evaluate model-context-protocol (MCP) based applications using `deepeval`, for both single-turn and multi-turn use cases.

## Overview

MCP evaluation is different from other evaluations because you can choose to create single-turn test cases or multi-turn test cases based on your application design and architecture.

**In this 10 min quickstart, you'll learn how to:**

- Track your MCP interactions
- Create test cases for your application
- Evaluate your MCP based application using MCP metrics

## Prerequisites

- Install `deepeval`
- A Confident AI API key (recommended). Sign up for one [here](https://app.confident-ai.com)

:::info
Confident AI allows you to view and share your testing reports. Set your API key in the CLI:
"""
logger.info("## Overview")

CONFIDENT_API_KEY="confident_us..."

"""
:::

## Understanding MCP Evals

**Model Context Protocol (MCP)** is an open-source framework developed by **Ollama** to standardize how AI systems, particularly large language models (LLMs), interact with external tools and data sources.
The MCP architecture is composed of three main components:

- **Host** — The AI application that coordinates and manages one or more MCP clients
- **Client** — Maintains a one-to-one connection with a server and retrieves context from it for the host to use
- **Server** — Paired with a single client, providing the context the client passes to the host

![MCP Architecture Image](https://deepeval-docs.s3.amazonaws.com/mcp-architecture.png)

`deepeval` allows you to evaluate the MCP host on various criterion like its primitive usage, argument generation and task completion.

## Run Your First MCP Eval

In `deepeval` MCP evaluations can be done using either single-turn or multi-turn test cases. In code, you'll have to track all MCP interactions and finally create a test case after the execution of your application.

<Timeline>
<TimelineItem title="Create an MCP server">

Connect your application to MCP servers and create the `MCPServer` object for all the MCP servers you're using.
"""
logger.info("## Understanding MCP Evals")


url = "https://example.com/mcp"

mcp_servers = []
tools_called = []

async def main():
    read, write, _  = await AsyncExitStack().enter_async_context(streamablehttp_client(url))
    logger.success(format_json(read, write, _))
    session = await AsyncExitStack().enter_async_context(ClientSession(read, write))
    logger.success(format_json(session))
    await session.initialize()

    tool_list = await session.list_tools()
    logger.success(format_json(tool_list))

    mcp_servers.append(MCPServer(
        name=url,
        transport="streamable-http",
        available_tools=tool_list.tools,
    ))

"""
</TimelineItem>
<TimelineItem title="Track your MCP interactions">

In your MCP application's main file, you need to track all the MCP interactions during run time. This includes adding `tools_called`, `resources_called` and `prompts_called` whenever your host uses them.

![MCP Interaction tracking](https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:evaluation-mcp-tools.png)
"""
logger.info("In your MCP application's main file, you need to track all the MCP interactions during run time. This includes adding `tools_called`, `resources_called` and `prompts_called` whenever your host uses them.")


available_tools = [
    {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
    for tool in tool_list
]

response = self.anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    tools=available_tools,
)

for content in response.content:
    if content.type == "tool_use":
        tool_name = content.name
        tool_args = content.input
        result = await session.call_tool(tool_name, tool_args)
        logger.success(format_json(result))

        tools_called.append(MCPToolCall(
            name=tool_name,
            args=tool_args,
            result=result
        ))

"""
You can also track any [resources](https://www.deepeval.com/docs/evaluation-mcp#resources) or [prompts](https://www.deepeval.com/docs/evaluation-mcp#prompts) if you use them. You are now tracking all the MCP interactions during run time of your application.

</TimelineItem>
<TimelineItem title="Create a test case">

You can now create a test case for your MCP application using the above interactions.
"""
logger.info("You can also track any [resources](https://www.deepeval.com/docs/evaluation-mcp#resources) or [prompts](https://www.deepeval.com/docs/evaluation-mcp#prompts) if you use them. You are now tracking all the MCP interactions during run time of your application.")

...

test_case = LLMTestCase(
    input=query,
    actual_output=response,
    mcp_servers=mcp_servers,
    mcp_tools_called=tools_called,
)

"""
The test cases must be created after the execution of your application. Click here to see a [full example on how to create single-turn test cases](https://github.com/confident-ai/deepeval/blob/main/examples/mcp_evaluation/mcp_eval_single_turn.py) for MCP evaluations.

:::tip
You can make your `main()` function return `mcp_servers`, `tools_called`, `resources_called` and `prompts_called`. This helps you import your MCP application anywhere and create test cases easily in different test files.
:::

</TimelineItem>
<TimelineItem title="Define metrics">

You can now use the [`MCPUseMetric`](/docs/metrics-mcp-use) to run evals on your single-turn your test case.
"""
logger.info("The test cases must be created after the execution of your application. Click here to see a [full example on how to create single-turn test cases](https://github.com/confident-ai/deepeval/blob/main/examples/mcp_evaluation/mcp_eval_single_turn.py) for MCP evaluations.")


mcp_use_metric = MCPUseMetric()

"""
</TimelineItem>
<TimelineItem title="Run an evaluation">

Run an evaluation on the test cases you previously created using the metrics defined above.
"""
logger.info("Run an evaluation on the test cases you previously created using the metrics defined above.")


evaluate([test_case], [mcp_use_metric])

"""
🎉🥳 **Congratulations!** You just ran your first single-turn MCP evaluation. Here's what happened:

- When you call `evaluate()`, `deepeval` runs all your `metrics` against all `test_cases`
- All `metrics` outputs a score between `0-1`, with a `threshold` defaulted to `0.5`
- The `MCPUseMetric` first evaluates your test case on its primitive usage to see how well your application has utilized the MCP capabilities given to it.
- It then evaluates the argument correctness to see if the inputs generated for your primitive usage were correct and accurate for the task.
- The `MCPUseMetric` then finally takes the minimum of the both scores to give a final score to your test case.

</TimelineItem>

<TimelineItem title="View on Confident AI (recommended)">

If you've set your `CONFIDENT_API_KEY`, test runs will appear automatically on [Confident AI](https://app.confident-ai.com), the DeepEval platform.

<VideoDisplayer 
    src="https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:getting-started-mcp-single-turn.mp4" 
    confidentUrl="https://www.confident-ai.com/docs/llm-evaluation/dashboards/testing-reports"
    label="Evaluations Test Reports on Confident AI"
/>

:::tip
If you haven't logged in, you can still upload the test run to Confident AI from local cache:
"""
logger.info("If you've set your `CONFIDENT_API_KEY`, test runs will appear automatically on [Confident AI](https://app.confident-ai.com), the DeepEval platform.")

deepeval view

"""
:::

</TimelineItem>

</Timeline>

## Multi-Turn MCP Evals

For multi-turn MCP evals, you are required to add the `mcp_tools_called`, `mcp_resource_called` and `mcp_prompts_called` in the `Turn` object for each turn of the assistant. (if any)

<Timeline>
<TimelineItem title="Track your MCP interactions">

During the interactive session of your application, you need to track all the MCP interactions. This includes adding `tools_called`, `resources_called` and `prompts_called` whenever your host uses them.

![MCP Interaction tracking](https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:evaluation-mcp-tools.png)
"""
logger.info("## Multi-Turn MCP Evals")


async def main():
    ...

    result = await session.call_tool(tool_name, tool_args)
    logger.success(format_json(result))
    tool_called = MCPToolCall(name=tool_name, args=tool_args, result=result)

    turns.append(
        Turn(
            role="assistant",
            content=f"Tool call: {tool_name} with args {tool_args}",
            mcp_tools_called=[tool_called],
        )
    )

"""
You can also track any [resources](https://www.deepeval.com/docs/evaluation-mcp#resources) or [prompts](https://www.deepeval.com/docs/evaluation-mcp#prompts) if you use them. You are now tracking all the MCP interactions during run time of your application.

</TimelineItem>
<TimelineItem title="Create a test case">

You can now create a test case for your MCP application using the above `turns` and `mcp_servers`.
"""
logger.info("You can also track any [resources](https://www.deepeval.com/docs/evaluation-mcp#resources) or [prompts](https://www.deepeval.com/docs/evaluation-mcp#prompts) if you use them. You are now tracking all the MCP interactions during run time of your application.")


convo_test_case = ConversationalTestCase(
    turns=turns,
    mcp_servers=mcp_servers
)

"""
The test cases must be created after the execution of the application. Click here to see a [full example on how to create multi-turn test cases](https://github.com/confident-ai/deepeval/blob/main/examples/mcp_evaluation/mcp_eval_multi_turn.py) for MCP evaluations.

:::tip
You can make your `main()` function return `turns` and `mcp_servers`. This helps you import your MCP application anywhere and create test cases easily in different test files.
:::

</TimelineItem>
<TimelineItem title="Define metrics">

You can now use the [MCP metrics](/docs/metrics-multi-turn-mcp-use) to run evals on your test cases. There's two metrics for multi-turn test cases that support MCP evals.
"""
logger.info("The test cases must be created after the execution of the application. Click here to see a [full example on how to create multi-turn test cases](https://github.com/confident-ai/deepeval/blob/main/examples/mcp_evaluation/mcp_eval_multi_turn.py) for MCP evaluations.")


mcp_use_metric = MultiTurnMCPUseMetric()
mcp_task_completion = MCPTaskCompletionMetric()

"""
</TimelineItem>
<TimelineItem title="Run an evaluation">

Run an evaluation on the test cases you previously created using the metrics defined above.
"""
logger.info("Run an evaluation on the test cases you previously created using the metrics defined above.")


evaluate([convo_test_case], [mcp_use_metric, mcp_task_completion])

"""
🎉🥳 **Congratulations!** You just ran your first multi-turn MCP evaluation. Here's what happened:

- When you call `evaluate()`, `deepeval` runs all your `metrics` against all `test_cases`
- All `metrics` outputs a score between `0-1`, with a `threshold` defaulted to `0.5`
- You used the `MultiTurnMCPUseMetric` and `MCPTaskCompletionMetric` for testing your MCP application
- The `MultiTurnMCPUseMetric` evaluates your application's capability on primitive usage and argument generation to get the final score.
- The `MCPTaskCompletionMetric` evaluates whether your application has satisfied the given task for all the interactions between user and assistant.

</TimelineItem>
<TimelineItem title="View on Confident AI (recommended)">

If you've set your `CONFIDENT_API_KEY`, test runs will appear automatically on [Confident AI](https://app.confident-ai.com), the DeepEval platform.

<VideoDisplayer 
    src="https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:getting-started-mcp-multi-turn.mp4" 
    confidentUrl="https://www.confident-ai.com/docs/llm-evaluation/multi-turn/end-to-end"
    label="Multi-Turn End-to-End Evals"
/>

:::tip
If you haven't logged in, you can still upload the test run to Confident AI from local cache:
"""
logger.info("If you've set your `CONFIDENT_API_KEY`, test runs will appear automatically on [Confident AI](https://app.confident-ai.com), the DeepEval platform.")

deepeval view

"""
:::

</TimelineItem>
</Timeline>

## Next Steps

Now that you have run your first MCP eval, you should:

1. **Customize your metrics**: You can change the threshold of your metrics to be more strict to your use-case.
2. **Prepare a dataset**: If you don't have one, [generate one](/docs/synthesizer-introduction) as a starting point to store your inputs as goldens.
3. **Setup Tracing**: If you created your own custom MCP server, you can [setup tracing](https://documentation.confident-ai.com/docs/llm-tracing/tracing-features/span-types) on your tool definitons.

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/llm-tracing:spans.mp4"
  confidentUrl="/docs/llm-tracing/introduction"
  label="Span-Level Evals in Production"
/>

You can [learn more about MCP here](/docs/evaluation-mcp).
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)