import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.agents.azure import AzureAIAgent
from azure.ai.agents.models import BingGroundingTool
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from jet.logger import CustomLogger
import dotenv
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Azure AI Foundry Agent

In AutoGen, you can build and deploy agents that are backed by the [Azure AI Foundry Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview) using the {py:class}`~autogen_ext.agents.azure._azure_ai_agent.AzureAIAgent` class. Here, important aspects of the agent including the provisioned model, tools (e.g, code interpreter, bing search grounding, file search etc.), observability, and security are managed by Azure. This allows you to focus on building your agent without worrying about the underlying infrastructure.

In this guide, we will explore an example of creating an Azure AI Foundry Agent using the `AzureAIAgent` that can address tasks using the Azure Grounding with Bing Search tool.
"""
logger.info("# Azure AI Foundry Agent")



"""
# Bing Search Grounding 

An {py:class}`~autogen_ext.agents.azure._azure_ai_agent.AzureAIAgent` can be assigned a set of tools including [Grounding with Bing Search](https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/bing-grounding?tabs=python&pivots=overview#setup). 

Grounding with Bing Search allows your Azure AI Agents to incorporate real-time public web data when generating responses. You need to create a Grounding with Bing Search resource, and then connect this resource to your Azure AI Agents. When a user sends a query, Azure AI Agents decide if Grounding with Bing Search should be leveraged or not. If so, it will leverage Bing to search over public web data and return relevant chunks. Lastly, Azure AI Agents will use returned chunks to generate a response.

## Prerequisites

- You need to have an Azure subscription.
- You need to have the Azure CLI installed and configured. (also login using the command `az login` to enable default credentials)
- You need to have the `autogen-ext[azure]` package installed.

You can create a [Grounding with Bing Search resource in the Azure portal](https://portal.azure.com/#create/Microsoft.BingGroundingSearch). Note that you will need to have owner or contributor role in your subscription or resource group to create it. Once you have created your resource, you can then pass it to the Azure Foundry Agent using the resource name.

In the following example, we will create a new Azure Foundry Agent that uses the Grounding with Bing Search resource.
"""
logger.info("# Bing Search Grounding")



dotenv.load_dotenv()


async def bing_example() -> None:
    async def async_func_14():
        async with DefaultAzureCredential() as credential:  # type: ignore
            async with AIProjectClient(  # type: ignore
                credential=credential, endpoint=os.getenv("AZURE_PROJECT_ENDPOINT", "")
            ) as project_client:
                async def run_async_code_c03899a1():
                    conn = await project_client.connections.get(name=os.getenv("BING_CONNECTION_NAME", ""))
                    return conn
                conn = asyncio.run(run_async_code_c03899a1())
                logger.success(format_json(conn))
            
                bing_tool = BingGroundingTool(conn.id)
                agent_with_bing_grounding = AzureAIAgent(
                    name="bing_agent",
                    description="An AI assistant with Bing grounding",
                    project_client=project_client,
                    deployment_name="gpt-4o",
                    instructions="You are a helpful assistant.",
                    tools=bing_tool.definitions,
                    metadata={"source": "AzureAIAgent"},
                )
            
            
                result = await agent_with_bing_grounding.on_messages(
                    messages=[
                        TextMessage(
                            content="What is Microsoft's annual leave policy? Provide citations for your answers.",
                            source="user",
                        )
                    ],
                    cancellation_token=CancellationToken(),
                    message_limit=5,
                )
                logger.debug(result)
            
            
        return result

    result = asyncio.run(async_func_14())
    logger.success(format_json(result))
async def run_async_code_68d3df3d():
    await bing_example()
    return 
 = asyncio.run(run_async_code_68d3df3d())
logger.success(format_json())

"""
Note that you can also provide other Azure Backed [tools](https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/overview) and local client side functions to the agent.

See the {py:class}`~autogen_ext.agents.azure._azure_ai_agent.AzureAIAgent` class api documentation for more details on how to create an Azure Foundry Agent.
"""
logger.info("Note that you can also provide other Azure Backed [tools](https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/overview) and local client side functions to the agent.")

logger.info("\n\n[DONE]", bright=True)