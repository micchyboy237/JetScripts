from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")






client = AzureAIChatCompletionClient(
    model="llama3.2",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

frontdesk_agent = AssistantAgent(
    "planner_agent",
    model_client=client,
    description="A helpful assistant that can plan trips.",
    system_message="""
    You are a Front Desk Travel Agent with ten years of experience and are known for brevity as you deal with many customers.
    The goal is to provide the best activities and locations for a traveler to visit.
    Only provide a single recommendation per response.
    You're laser focused on the goal at hand.
    Don't waste time with chit chat.
    Consider suggestions when refining an idea.""",
)

concierge_agent = AssistantAgent(
    "concierge_agent",
    model_client=client,
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="""
    You are an are hotel concierge who has opinions about providing the most local and authentic experiences for travelers.
    The goal is to determine if the front desk travel agent has recommended the best non-touristy experience for a traveler.
    If so, respond with 'APPROVE'
    If not, provide insight on how to refine the recommendation without using a specific example.
    """,
)

termination = TextMentionTermination("APPROVE")
team = RoundRobinGroupChat(
    [frontdesk_agent, concierge_agent], termination_condition=termination
)

async for message in team.run_stream(task="I would like to plan a trip to Paris."):
    if isinstance(message, TaskResult):
        logger.debug("Stop Reason:", message.stop_reason)
    else:
        logger.debug(message)

logger.info("\n\n[DONE]", bright=True)