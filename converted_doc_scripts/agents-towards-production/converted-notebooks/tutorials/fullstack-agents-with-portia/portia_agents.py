from dotenv import load_dotenv
from jet.logger import logger
from pathlib import Path
from portia import (
Tool,
ToolHardError,
ToolRunContext,
)
from portia import (Config, Portia, DefaultToolRegistry)
from portia import (PlanRun, Step, Output, Tool)
from portia.cli import CLIExecutionHooks
from portia.model import Message
from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List
import glob
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
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--fullstack-agents-with-portia--portia-agents)

# Portia AI Framework Tutorial: Building Production-Ready Agentic Workflows
## Overview

This comprehensive tutorial demonstrates how to leverage [Portia AI](https://portiaai.org/3J8gAwY), an open-source developer framework designed for creating predictable, stateful, and authenticated agentic workflows. Portia AI empowers developers to maintain precise control over their multi-agent deployments while ensuring production readiness through robust state management and authentication mechanisms.

Throughout this tutorial, we will construct an intelligent agent system that analyzes user experience research (UXR) data and automatically organizes the findings within Notion. This practical example showcases the framework's capabilities in handling real-world data processing and integration tasks.

## What You Will Learn

This tutorial is structured as a progressive journey through six fundamental concepts that form the backbone of effective agentic workflow development. Each section builds upon the previous one, creating a comprehensive understanding of the Portia AI framework.

We begin with basic planning mechanisms and progress through advanced features like custom tool integration, structured data outputs, cloud service authentication, and sophisticated execution control through custom hooks. By the end of this tutorial, you will have hands-on experience with production-grade multi-agent system development.

## Architecture Overview

The Portia AI framework operates on a sophisticated multi-layered architecture that separates concerns between planning, execution, and state management. The system employs dedicated agents for different phases of workflow execution, ensuring robust error handling and predictable behavior patterns.

```mermaid
graph TD
    A[Planning Agent] --> B[Execution Agents]
    B --> C[State Management]
    D[Tool Registry] --> A 
    B --> D
    
    A1[Query Analysis] --> A
    A2[Step Planning] --> A
    A3[Tool Selection] --> A
    
    B1[Tool Calls] --> B
    B2[Output Extract] --> B
    
    C1[Step Tracking] --> C
    C2[Error Handling] --> C
    C3[Authentication] --> C
    
    D1[Local Tools] --> D
    D2[Custom Tools] --> D
    D3[Cloud MCP Servers] --> D
    D4[Authentication Layer] --> D
```

The planning agent analyzes user queries and constructs multi-step execution plans with appropriate tool selections. Execution agents then carry out these plans while maintaining stateful information about progress, outputs, and any encountered issues. The tool registry provides a unified interface for both local custom tools and remote cloud services, with built-in authentication management.

PS: This directory includes a "uxr" folder with some sample files we will read from during the tutorial.

## Prerequisites and Environment Setup

Before diving into the core concepts, we need to establish the proper development environment and configure the necessary authentication credentials. The environment setup process involves installing the Portia SDK and configuring your preferred large language model provider. Portia AI supports multiple LLM providers, allowing you to choose the most suitable option for your specific use case and budget requirements.
"""
logger.info("# Portia AI Framework Tutorial: Building Production-Ready Agentic Workflows")

# !pip install portia-sdk-python




load_dotenv(override=True)

"""
# 1. Fundamental Planning Architecture

## Understanding Multi-Step Plan Generation

The foundation of any effective agentic workflow lies in intelligent planning. Portia AI's planning system analyzes complex user queries and decomposes them into manageable, sequential steps that can be executed by specialized agents. This approach ensures that even sophisticated multi-step operations can be handled reliably and predictably.

When you invoke the planning agent through the `portia.plan` method, the system performs several critical operations behind the scenes. First, it analyzes the user query to understand the intent and identify the required capabilities. Then, it examines the available tool registry to determine which tools can best accomplish each sub-task. Finally, it constructs a comprehensive execution plan that includes proper sequencing, input/output flow, and error handling strategies.

The resulting plan object contains detailed information about each execution step, including the specific tools that will be used, the expected inputs and outputs, and the logical flow between steps. This granular planning approach allows for better debugging, monitoring, and optimization of agent workflows.
"""
logger.info("# 1. Fundamental Planning Architecture")



path = "./uxr/calorify.txt"
query =f"Read the user feedback notes in local file {path}, \
            and call out recurring themes in their feedback."

config = Config.from_default()
portia = Portia(
    config=config,
    tools=DefaultToolRegistry(config=config)
)

plan = portia.plan(query=query)
logger.debug(f"{plan.model_dump_json(indent=2)}")

"""
# 2. Stateful Execution and Progress Tracking

## Managing Complex Workflow State

One of Portia AI's most powerful features is its ability to maintain comprehensive state information throughout the execution of multi-step plans. This stateful approach ensures that complex workflows can be monitored, debugged, and even resumed if interruptions occur. The execution system tracks not only the current progress but also maintains detailed logs of all tool calls, outputs, and any encountered issues.

The `portia.run_plan` method serves as the orchestrator for this complex execution process. It systematically works through each step in the plan, invokes the appropriate execution agents, manages tool calls, and extracts relevant outputs that feed into subsequent steps. This approach creates a robust execution environment where each step builds upon the results of previous operations.

The plan run state object that results from this execution contains a wealth of information that proves invaluable for both development and production deployment. It includes the current execution step, detailed outputs from each completed step, timing information, and comprehensive error logs if any issues were encountered.
"""
logger.info("# 2. Stateful Execution and Progress Tracking")

plan_run = portia.run_plan(plan)
logger.debug(f"{plan_run.model_dump_json(indent=2)}")

"""
# 3. Custom Tool Development and Integration

## Extending Framework Capabilities with Specialized Tools

While Portia AI provides a comprehensive set of built-in tools through its DefaultToolRegistry, real-world applications often require specialized functionality that goes beyond standard operations. The framework's extensible architecture allows developers to create custom tools that integrate seamlessly with the existing ecosystem while maintaining the same reliability and error handling standards.

Custom tool development follows a structured approach that ensures consistency and reliability across all tool implementations. Each tool must define a clear input schema using Pydantic models, implement a standardized run method, and provide appropriate error handling mechanisms. This standardization ensures that custom tools work seamlessly with the framework's planning and execution systems.

The error handling strategy employed in custom tools demonstrates an important principle in Portia AI development. The `ToolHardError` exception signals to the execution agents that a critical failure has occurred and that the plan run should be terminated rather than attempting recovery.
"""
logger.info("# 3. Custom Tool Development and Integration")



class FileSearchToolSchema(BaseModel):
    """
    Input schema for the FileSearchTool.
    """

    directory_path: str = Field(
        ..., description="The directory path to search for files in"
    )


class FileSearchTool(Tool[str]):
    """
    Tool for searching files in a local directory.
    """

    id: str = "file_search_tool"
    name: str = "File Search Tool"
    description: str = """
    Searches for all files in a local directory recursively.
    """
    args_schema: type[BaseModel] = FileSearchToolSchema
    output_schema: tuple[str, str] = (
        "list",
        "A list of file paths found in the directory",
    )

    def run(
        self,
        ctx: ToolRunContext,
        directory_path: str,
    ) -> List[str]:
        """
        Run the File Search Tool to find all files in the directory.

        Args:
            ctx: The tool run context
            directory_path: Directory to search in

        Returns:
            List of all file paths found in the directory
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ToolHardError(f"Directory does not exist: {directory_path}")

        if not dir_path.is_dir():
            raise ToolHardError(f"Path is not a directory: {directory_path}")

        matching_files = []
        try:
            search_pattern = str(dir_path / "**" / "*")
            for file_path in glob.glob(search_pattern, recursive=True):
                path_obj = Path(file_path)

                if not path_obj.is_file():
                    continue

                matching_files.append(str(path_obj.absolute()))

        except Exception as e:
            raise ToolHardError(f"Error searching for files: {str(e)}")

        matching_files.sort()

        return matching_files

"""
# 4. Structured Output Management

## Ensuring Consistent Data Formats Through Schema Definition

As agentic workflows become more sophisticated, the need for consistent, predictable output formats becomes paramount. Structured outputs serve multiple purposes in production systems: they ensure data consistency across different execution runs, enable easier integration with downstream systems, and provide clear contracts that other parts of your application can rely upon.

Portia AI's structured output system leverages Pydantic models to define exact schemas that the framework will enforce during execution. This approach combines the flexibility of natural language processing with the reliability of strongly-typed data structures, creating a powerful hybrid that works well in production environments.

The integration of custom tools with structured outputs demonstrates the composability that makes Portia AI particularly powerful for complex workflows. The planning system automatically understands how to coordinate between the custom FileSearchTool and the structured output requirements, creating execution plans that leverage both capabilities seamlessly.

The `portia.run` method represents a streamlined approach that combines planning and execution in a single operation. This convenience method is particularly useful during development and for simpler workflows, while still maintaining all the robustness and state management capabilities.
"""
logger.info("# 4. Structured Output Management")

path = "./uxr"
query =f"Recap the user feedback notes for each file I have in the folder {path}, \
            and call out pros and cons in their feedback."

class UserFeedback(BaseModel):
    """A summary of a user's feedback"""
    user_name: str = Field(..., description="The name of the user")
    feedback_summary: str = Field(..., description="A summary of the user's feedback")

class UserFeedbackList(BaseModel):
    """A list of feedback from multiple users"""
    feedback: list[UserFeedback] = Field(..., description="A list of all user feedback summaries")

tools = DefaultToolRegistry(config=config) + [FileSearchTool()]
portia = Portia(
    config=config,
    tools=tools,
)

plan_run = portia.run(
    query=query,
    structured_output_schema=UserFeedbackList
)

for feedback_item in plan_run.outputs.final_output.value.feedback:
    logger.debug(feedback_item.model_dump_json(indent=2))

"""
# 5. Cloud Integration and Authentication Management

## Seamlessly Connecting to Remote Services with Built-In Security

Modern agentic workflows frequently need to interact with cloud services and external APIs. Managing authentication for these services traditionally represents one of the most complex aspects of building production-grade automation systems. Portia AI addresses this challenge through its cloud platform and MCP (Model Context Protocol) server integration, which provides secure, managed authentication for over 1000 different tools and services.

Portia cloud offers the ability to store plan runs and tool call logs in the cloud. It also includes 1000+ cloud and MCP tools with built-in auth. You first need to obtain a Portia API key -- Head over to our dashboard [here](https://app.portialabs.ai/dashboard?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial
) and navigate to the `Manage API keys` tab from the left hand nav. There you can generate a new API key. We offer a generous free plan, so you'll be able to complete this entire tutorial without upgrading.

Once you have an API key, your `DefaultToolRegistry` now includes a bunch of tool definitions exposed by Portia cloud. You can get acquainted with those from the [docs](https://docs.portialabs.ai/portia-tools?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial). You can also view them from the dashboard's `Tool Registry` tab. The next thing you will need to do is add the Notion remote MCP server to your default tool registry from there. You will be asked for a one-off authentication *as the admin* setting up the MCP server connection in order to access all tool definitions (a strange quirk of the MCP standard at the moment). This will not have any bearing on which Notion projects your end users will have access to. We handle authentication for them individually as you will see below.

💡 Note that Portia stores all authentication credentials using [production-grade encryption](https://docs.portialabs.ai/security?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial).

<img src="./notion.gif" controls autoplay muted style="width: 75%; border-radius: 8px; margin: 24px 0;"></img>
"""
logger.info("# 5. Cloud Integration and Authentication Management")

[logger.debug(f"Tool name: {tool.name}\nDescription: {tool.description}\n")
for tool in tools.get_tools() if 'notion' in tool.id]

"""
## Implementing Cloud Service Integration

If you passed the check above and can now see the Notion tools you are ready for the next step. We're going to change our query once again, this time asking Portia to publish the UXR feedback pulled from the local files onto a new Notion page. Notice how the plan now includes a new step with the correct Notion tool from their MCP server. Because Portia now has access to the Notion MCP server it was able to include them in its planning process seamlessly.

We're also going to introduce `ExecutionHooks` for the first time. These abstractions are used to define when and how to interrupt execution agents during a plan run in order to solicit human input. In the implementation below we will raise a request for the person doing this multi-agent run to authentication into Notion and pick a workspace so our agents can be armed with the necessary oauth token to complete their job.

⚠️ **Look out for the `Clarification` object raised at the step where `portia:mcp:mcp.notion.com:notion_create_pages` tool. You notice an oauth URL you need to click on in order to authenticate the agent into your account and workspace e.g. `https://mcp.notion.com/authorize?redirect_uri=https%3A%2F%2Fapi.portialabs.ai...`**
"""
logger.info("## Implementing Cloud Service Integration")


query = f"Read the user feedback notes for each file I have in the folder {path}, \
            and call out pros and cons in their feedback.\
            Finally publish the output on a new notion page titled 'UXR Summary'"

portia = Portia(
    config=config,
    tools=tools,
    execution_hooks=CLIExecutionHooks()
)

plan = portia.plan(query=query)
logger.debug(plan.model_dump_json(indent=2))

plan_run = portia.run_plan(plan)
logger.debug(plan_run.model_dump_json(indent=2))

"""
Now is a good time to get acquainted with more features in the Portia dashboard. You've earned it after all this squinting at the command line on here. Head over the to `Plan Runs` tab and find the latest plan run. As long as your API key is active all plan runs and tool call logs moving forward will be stored in real-time in the cloud so you can watch future plan runs progress from the dashboard directly!

# 6. Advanced Control Through Custom Execution Hooks

## Implementing Sophisticated Workflow Control and Quality Assurance

The final piece of the Portia AI framework puzzle involves custom execution hooks, which provide powerful mechanisms for implementing quality assurance, content filtering, and sophisticated workflow control. These hooks allow you to insert custom logic at specific points during plan execution, enabling real-time monitoring and intervention capabilities that are essential for production deployments.

Execution hooks operate at various stages of the workflow execution process. They can be configured to run before tool calls, after tool calls, or at other significant execution milestones. This flexibility enables a wide range of use cases, from simple logging and monitoring to complex content validation and business rule enforcement.

Now let's use a custom `ExecutionHooks` object to filter for profanity and fail a plan run before any colourful language makes into a plan run state. We will create a `profanity_check_hook` method as a `Callable` subclass that gets invoked after every tool call (and before the output of the tool is ingested into the plan run state). In that method we're going to use one of the LLMs that Portia is currently configured to rely on (the "introspection model" is responsible for conditionals etc. See model override options for [here](https://docs.portialabs.ai/manage-config#model-overrides?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial)), and request it to scan the tool call output at the current step for profanity. If it does detect any profanity a `ToolHardError` is raised which will cause the plan run to exit with a FAILED state. To trigger this exception make sure one of your UXR file now includes some of your favourite cussing. I certainly went to town with it!

Why don't you kick off the plan run below and hop over to the dashboard to monitor your plan's progress. You should see an error during one of the file read operations per screenshot below.

<img src="./profanity.png" controls autoplay muted style=" border-radius: 8px; margin: 24px 0;"></img>
"""
logger.info("# 6. Advanced Control Through Custom Execution Hooks")


class ProfanityResult(BaseModel):
    contains_profanity: bool

def profanity_check_hook(
    tool: Tool,
    output: Output,
    plan_run: PlanRun,
    step: Step,
):
    profanity_result = config.get_introspection_model().get_structured_response(
        messages = [Message(
            role="user",
            content=f"""Given the output of the current tool call as shown below, if the user input contains
            any profanity, return 1.\n{output}""")],
        schema=ProfanityResult
    )

    logger.debug(f"\n***** Profanity filter at step #{plan_run.current_step_index} is {profanity_result.contains_profanity} *****\n")
    if profanity_result.contains_profanity:
        raise ToolHardError(f"Interrupted plan run due profanity detection at step #{plan_run.current_step_index}")

portia = Portia(
    config=config,
    tools=tools,
    execution_hooks=CLIExecutionHooks(
            after_tool_call=profanity_check_hook
    )
)

plan_run = portia.run_plan(plan)
logger.debug(plan_run.model_dump_json(indent=2))

logger.info("\n\n[DONE]", bright=True)