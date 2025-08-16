import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OllamaChatCompletionClient
from autogenstudio.database import DatabaseManager
from autogenstudio.teammanager import TeamManager
from jet.logger import CustomLogger
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## AutoGen Studio Agent Workflow API Example

This notebook focuses on demonstrating capabilities of the autogen studio workflow python api.  

- Declarative Specification of an Agent Team
- Loading the specification and running the resulting agent
"""
logger.info("## AutoGen Studio Agent Workflow API Example")


wm = TeamManager()
async def run_async_code_c761f032():
    async def run_async_code_6499fc36():
        result = await wm.run(task="What is the weather in New York?", team_config="team.json")
        return result
    result = asyncio.run(run_async_code_6499fc36())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_c761f032())
logger.success(format_json(result))
logger.debug(result)

result_stream =  wm.run_stream(task="What is the weather in New York?", team_config="team.json")
async for response in result_stream:
    logger.debug(response)

"""
## Load Directly Using the AgentChat API
"""
logger.info("## Load Directly Using the AgentChat API")

team_config = json.load(open("team.json"))
team = BaseGroupChat.load_component(team_config)
logger.debug(team._participants)

"""
## AutoGen Studio Database API

Api for creating objects and serializing to a database.
"""
logger.info("## AutoGen Studio Database API")


os.makedirs("test", exist_ok=True)
dbmanager = DatabaseManager(engine_uri="sqlite:///test.db", base_dir="test")
dbmanager.initialize_database()

"""
## Sample AgentChat Example (Python)
"""
logger.info("## Sample AgentChat Example (Python)")


planner_agent = AssistantAgent(
    "planner_agent",
    model_client=OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096),
    description="A helpful assistant that can plan trips.",
    system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request. Respond with a single sentence",
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096),
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided. Respond with a single sentence",
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096),
    description="A helpful assistant that can provide language tips for a given destination.",
    system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.Respond with a single sentence",
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096),
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed tfinal travel plan. You must ensure th b at the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.Respond with a single sentence",
)

termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
group_chat = RoundRobinGroupChat(
    [planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination
)

result = group_chat.run_stream(task="Plan a 3 day trip to Nepal.")
async for response in result:
    logger.debug(response)


config = group_chat.dump_component().model_dump()

with open("travel_team.json", "w") as f:
    json.dump(config, f, indent=4)

with open("travel_team.json", "r") as f:
    config = json.load(f)

group_chat = RoundRobinGroupChat.load_component(config)
result = group_chat.run_stream(task="Plan a 3 day trip to Nepal.")
async for response in result:
    logger.debug(response)

logger.info("\n\n[DONE]", bright=True)