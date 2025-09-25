from jet.transformers.formatters import format_json
from deepeval import evaluate
from deepeval.metrics import MCPUseMetric
from deepeval.metrics import MultiTurnMCPMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import LLMTestCase
from deepeval.test_case import MCPPromptCall
from deepeval.test_case import MCPResourceCall
from deepeval.test_case import MCPServer
from deepeval.test_case import MCPToolCall
from deepeval.test_case.mcp import (
MCPServer,
MCPToolCall,
MCPResourceCall,
MCPPromptCall
)
from deepeval.test_case.mcp import MCPServer
from jet.logger import logger
from mcp import ClientSession
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
---
id: evaluation-mcp
title: Model Context Protocol (MCP)
sidebar_label: MCP
---

**Model Context Protocol (MCP)** is an open-source framework developed by **Ollama** to standardize how AI systems, particularly large language models (LLMs), interact with external tools and data sources.

## Architecture

The MCP architecture is composed of three main components:

- **Host** – The AI application that coordinates and manages one or more MCP clients.
- **Client** – Maintains a one-to-one connection with a server and retrieves context from it for the host to use.
- **Server** – Paired with a single client, providing the context the client passes to the host.

![MCP Architecture Image](https://deepeval-docs.s3.amazonaws.com/mcp-architecture.png)

For example, Claude acts as the MCP host. When Claude connects to an MCP server such as Google Sheets, the Claude runtime instantiates an MCP client that maintains a dedicated connection to that server. When Claude subsequently connects to another MCP server, such as Google Docs, it instantiates an additional MCP client to maintain that second connection. This preserves a one-to-one relationship between MCP clients and MCP servers, with the host (Claude) orchestrating multiple clients.

## Primitives

`deepeval` adheres to MCP primitives. You'll need to use these primitives to create an `MCPServer` class in `deepeval` before evaluation.

There are three core primitives that MCP servers can expose:

- **Tools**: Executable functions that LLM apps can invoke to perform actions
- **Resources**: Data sources that provide contextual information to LLM apps
- **Prompts**: Reusable templates that help structure interactions with language models

You can get all three primitives from `mcp`'s `ClientSession`:
"""
logger.info("## Architecture")


session = ClientSession(...)

tool_list = await session.list_tools()
logger.success(format_json(tool_list))
resource_list = await session.list_resources()
logger.success(format_json(resource_list))
prompt_list = await session.list_prompts()
logger.success(format_json(prompt_list))

"""
:::info
It is the MCP **server developer's** job to expose these primitives for you to leverage for evaluation. This means that you might not always have control over the MCP server you're interacting with.
:::

## MCP Server

The `MCPServer` class is an abstraction **provided by `deepeval`** to contain information about different MCP servers and the primitives they provide which can be used during evaluations.

Here's how how to create a `MCPServer` instance:
"""
logger.info("## MCP Server")


mcp_server = MCPServer(
    server_name="GitHub",
    transport="stdio",
    available_tools=tool_list.tools, # get from ClientSession
    available_resources=resource_list.resources, # get from ClientSession
    available_prompts=prompt_list.prompts # get from ClientSession
)

"""
The `MCPServer` accepts **FIVE** parameters:

- `server_name`: an optional string you can provide to store details about your MCP server.
- [Optional] `transport`: an optional literal that stores on the type of transport your MCP server uses. This information does not affect the evaluation of your MCP test case.
- [Optional] `available_tools`: an optional list of tools that your MCP server enables you to use.
- [Optional] `available_prompts`: an optional list of prompts that your MCP server enables you to use.
- [Optional] `available_resources`: an optional list of resources that your MCP server enables you to use.

:::tip
You need to make sure to provide the `.tools`, `.resources` and `.prompts` from the `list` method's response. They are each of type `Tool`, `Resource` and `Prompt` respectively from `mcp.types` and they are standardized from the official [MCP python sdk](https://github.com/modelcontextprotocol/python-sdk).
:::

## MCP At Runtime

During runtime, you'll inevitably be calling your MCP server which will then invoke tools, prompts, and resources. To run evaluation on MCP powered LLM apps, you'll need to format each of these primitives that were called for a given input.

### Tools

Provide a list of `MCPToolCall` objects for every tool your agent invokes during the interaction. The example below shows invoking a tool and constructing the corresponding `MCPToolCall`:
"""
logger.info("## MCP At Runtime")


session = ClientSession(...)

tool_name = "..."
tool_args = "..."

result = await session.call_tool(tool_name, tool_args)
logger.success(format_json(result))

mcp_tool_called = MCPToolCall(
    name=tool_name,
    args=tool_args,
    result=result,
)

"""
The `result` returned by `session.call_tool()` is a `CallToolResult` from `mcp.types`.

### Resources

Provide a list of `MCPResourceCall` objects for every resource your agent reads. The example below shows reading a resource and constructing the corresponding `MCPResourceCall`:
"""
logger.info("### Resources")


session = ClientSession(...)

uri = "..."

result = await session.read_resource(uri)
logger.success(format_json(result))

mcp_resource_called = MCPResourceCall(
    uri=uri,
    result=result,
)

"""
The `result` returned by `session.read_resource()` is a `ReadResourceResult` from `mcp.types`.

### Prompts

Provide a list of `MCPPromptCall` objects for every prompt your agent retrieves. The example below shows fetching a prompt and constructing the corresponding `MCPPromptCall`:
"""
logger.info("### Prompts")


session = ClientSession(...)

prompt_name = "..."

result = await session.get_prompt(prompt_name)
logger.success(format_json(result))

mcp_prompt_called = MCPPromptCall(
    name=prompt_name,
    result=result,
)

"""
The `result` returned by `session.get_prompt()` is a `GetPromptResult` from `mcp.types`.

## Evaluating MCP

You can evaluate MCPs for both **single and multi-turn** use cases. Evaluating MCP involves 4 steps:

- Defining an `MCPServer`, and
- Piping runtime primitives data into `deepeval`
- Creating a single-turn or multi-turn test case using these data
- Running MCP metrics on the test cases you've defined

### Single-Turn

The [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case) is a single-turn test case and accepts the following optional parameters to support MCP evaluations:
"""
logger.info("## Evaluating MCP")


test_case = LLMTestCase(
    input="...", # Your input
    actual_output="..." # Your LLM app's output
    mcp_servers=[MCPServer(...)],
    mcp_tools_called=[MCPToolCall(...)],
    mcp_prompts_called=[MCPPromptCall(...)],
    mcp_resources_called=[MCPResourceCall(...)]
)

evaluate(test_cases=[test_case], metrics=[MCPUseMetric])

"""
Typically all MCP parameters in a test case is optional. However if you wish to use MCP metrics such as the `MCPUseMetric`, you'll have to provide some of the following:

- `mcp_servers` — a list of `MCPServer`s
- `mcp_tools_called` — a list of `MCPToolCall` objects that your LLM app has used
- `mcp_resources_called` — a list of `MCPResourceCall` objects that your LLM app has used
- `mcp_prompts_called` — a list of `MCPPromptCall` objects that your LLM app has used

You can learn more about the `MCPUseMetric` [here.](/docs/metrics-mcp-use)

### Multi-Turn

The [`ConversationalTestCase`](/docs/evaluation-multiturn-test-cases#conversational-test-case) accepts an optional parameter called `mcp_server` to add your `MCPServer` instances, which tells `deepeval` how your MCP interactions should be evaluated:
"""
logger.info("### Multi-Turn")


test_case = ConversationalTestCase(
    turns=turns,
    mcp_servers=[MCPServer(...), MCPServer(...)]
)

evaluate(test_cases=[test_case], metrics=[MultiTurnMCPMetric()])

"""
<details>

<summary>
  Click here to see how to set MCP primitives for turns at runtime
</summary>

To set primitives at runtime, the `Turn` object accepts optional parameters like `mcp_tools_called`, `mcp_resources_called` and `mcp_prompts_called`, just like in an `LLMTestCase`:
"""
logger.info("Click here to see how to set MCP primitives for turns at runtime")


turns = [
    Turn(role="user", content="Some example input"),
    Turn(
        role="assistant",
        content="Do this too", # Your content here for a tool / resource / prompt call
        mcp_tools_called=[MCPToolCall(...)],
        mcp_resources_called=[MCPResourceCall(...)],
        mcp_prompts_called=[MCPPromptCall(...)],
    )
]

test_case = ConversationalTestCase(
    turns=turns,
    mcp_servers=[MCPServer(...)],
)

"""
</details>

✅ Done. You can now use the [MCP metrics](/docs/metrics-multi-turn-mcp-use) to run evaluations on your MCP based application.
"""

logger.info("\n\n[DONE]", bright=True)