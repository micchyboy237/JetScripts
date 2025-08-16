import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from jet.logger import CustomLogger
from typing import Any, Callable, Set, Dict, List, Optional
from typing import Dict, List, Optional
import json
import os
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen Tool Use Example

## Import the Needed Packages
"""
logger.info("# AutoGen Tool Use Example")



"""
## Creating the Client 

In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 

The `model` is defined as `llama3.1`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 

As a quick test, we will just run a simple prompt - `What is the capital of France`.
"""
logger.info("## Creating the Client")

client = AzureAIChatCompletionClient(
    model="llama3.1",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

async def run_async_code_2defc511():
    async def run_async_code_eab808f6():
        result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_eab808f6())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_2defc511())
logger.success(format_json(result))
logger.debug(result)

"""
## Defining the Functions 

In this example, we will give the agent access to a tool that is a function with a list of available vacation destinations and their availability. 

You can think that this would be a scenario where a travel agent might have an access to a travel database for example. 

As you go through this sample, feel free to try to define new functions and tools that the agent can call.
"""
logger.info("## Defining the Functions")



def vacation_destinations(city: str) -> tuple[str, str]:
    """
    Checks if a specific vacation destination is available

    Args:
        city (str): Name of the city to check

    Returns:
        tuple: Contains city name and availability status ('Available' or 'Unavailable')
    """
    destinations = {
        "Barcelona": "Available",
        "Tokyo": "Unavailable",
        "Cape Town": "Available",
        "Vancouver": "Available",
        "Dubai": "Unavailable",
    }

    if city in destinations:
        return city, destinations[city]
    else:
        return city, "City not found"

"""
## Defining the Function Tool 
To have the agent use the `vacation_destinations` as a `FunctionTool`, we need to define it as one. 

We will also provide a description of the to tool which is helpful for the agent to identify what that tool is used for in relation to the task the user has requested.
"""
logger.info("## Defining the Function Tool")

get_vacations = FunctionTool(
    vacation_destinations, description="Search for vacation destinations and if they are available or not."
)

"""
## Defining the agent 

Now we can create the agent in the below code. We define the `system_message` to give the agent instructions on how to help users find vacation destinations. 

We also set the `reflect_on_tool_use` parameter to true. This allows use the LLM to take the response of the tool call and send the response using natural language. 

You can set the the parameter to false to see the difference.
"""
logger.info("## Defining the agent")

agent = AssistantAgent(
    name="assistant",
    model_client=client,
    tools=[get_vacations],
    system_message="You are a travel agent that helps users find vacation destinations.",
    reflect_on_tool_use=True,
)

"""
## Running the Agent 

Now we can run the agent with the initial user message asking to take a trip to Tokyo. 

You can change this city desintation to see how the agent responds to the availablity of the city.
"""
logger.info("## Running the Agent")

async def assistant_run() -> None:
    async def async_func_1():
        response = await agent.on_messages(
            [TextMessage(content="I would like to take a trip to Tokyo", source="user")],
            cancellation_token=CancellationToken(),
        )
        return response
    response = asyncio.run(async_func_1())
    logger.success(format_json(response))
    logger.debug(response.inner_messages)
    logger.debug(response.chat_message)


async def run_async_code_1a8d04aa():
    await assistant_run()
    return 
 = asyncio.run(run_async_code_1a8d04aa())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)