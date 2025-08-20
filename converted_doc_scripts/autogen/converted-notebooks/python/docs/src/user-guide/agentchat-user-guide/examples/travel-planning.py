import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Travel Planning 

In this example, we'll walk through the process of creating a sophisticated travel planning system using AgentChat. Our travel planner will utilize multiple AI agents, each with a specific role, to collaboratively create a comprehensive travel itinerary.  

First, let us import the necessary modules.
"""
logger.info("# Travel Planning")


"""
### Defining Agents 

In the next section we will define the agents that will be used in the travel planning team.
"""
logger.info("### Defining Agents")

model_client = MLXAutogenChatLLMAdapter(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats")

planner_agent = AssistantAgent(
    "planner_agent",
    model_client=model_client,
    description="A helpful assistant that can plan trips.",
    system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=model_client,
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=model_client,
    description="A helpful assistant that can provide language tips for a given destination.",
    system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=model_client,
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
)

termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [planner_agent, local_agent, language_agent,
        travel_summary_agent], termination_condition=termination
)


async def run_async_code_21ea603f():
    await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))
    return
asyncio.run(run_async_code_21ea603f())


async def run_async_code_0349fda4():
    await model_client.close()
    return
asyncio.run(run_async_code_0349fda4())

logger.info("\n\n[DONE]", bright=True)
