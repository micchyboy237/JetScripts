import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, HTML
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen Basic Sample 

In this code sample, you will use the [AutoGen](https://aka.ms/ai-agents/autogen) AI Framework to create a basic agent. 

The goal of this sample is to show you the steps that we will later use in the additional code samples when implementing the different agentic patterns.

## Import the Needed Python Packages
"""
logger.info("# AutoGen Basic Sample")




"""
## Create the Client 

In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 

The `model` is defined as `llama3.1`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 

As a quick test, we will just run a simple prompt - `What is the capital of France`.
"""
logger.info("## Create the Client")

load_dotenv()
client = AzureAIChatCompletionClient(
    model="llama3.1",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
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
## Defining the Agent 

Now that we have set up the `client` and confirmed that it is working, let us create an `AssistantAgent`. Each agent can be assigned a: 
**name** - A short hand name that will be useful in referencing it in multi-agent flows. 
**model_client** - The client that you created in the earlier step. 
**tools** - Available tools that the Agent can use to complete a task.
**system_message** - The metaprompt that defines the task, behavior and tone of the LLM. 

You can change the system message to see how the LLM responds. We will cover `tools` in Lesson #4.
"""
logger.info("## Defining the Agent")

agent = AssistantAgent(
    name="assistant",
    model_client=client,
    tools=[],
    system_message="You are a travel agent that plans great vacations",
)

"""
## Run the Agent 

The below function will run the agent. We use the the `on_message` method to update the Agent's state with the new message. 

In this case, we update the state with a new message from the user which is `"Plan me a great sunny vacation"`.

You can change the message content to see how the LLM responds differently.
"""
logger.info("## Run the Agent")



async def assistant_run():
    user_query = "Plan me a great sunny vacation"

    html_output = "<div style='margin-bottom:10px'>"
    html_output += "<div style='font-weight:bold'>User:</div>"
    html_output += f"<div style='margin-left:20px'>{user_query}</div>"
    html_output += "</div>"

    async def async_func_11():
        response = await agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        return response
    response = asyncio.run(async_func_11())
    logger.success(format_json(response))

    html_output += "<div style='margin-bottom:20px'>"
    html_output += "<div style='font-weight:bold'>Assistant:</div>"
    html_output += f"<div style='margin-left:20px; white-space:pre-wrap'>{response.chat_message.content}</div>"
    html_output += "</div>"

    display(HTML(html_output))

async def run_async_code_1a8d04aa():
    await assistant_run()
    return 
 = asyncio.run(run_async_code_1a8d04aa())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)