import asyncio
from jet.transformers.formatters import format_json
from autogen_core.models import UserMessage, SystemMessage, AssistantMessage
from autogen_ext.models.openai import AzureOllamaChatCompletionClient
from enum import Enum
from jet.logger import CustomLogger
from pprint import pprint
from pydantic import BaseModel
from typing import List, Optional, Union
from typing import Optional
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


class AgentEnum(str, Enum):
    FlightBooking = "flight_booking"
    HotelBooking = "hotel_booking"
    CarRental = "car_rental"
    ActivitiesBooking = "activities_booking"
    DestinationInfo = "destination_info"
    DefaultAgent = "default_agent"
    GroupChatManager = "group_chat_manager"

class TravelSubTask(BaseModel):
    task_details: str
    assigned_agent: AgentEnum # we want to assign the task to the agent

class TravelPlan(BaseModel):
    main_task: str
    subtasks: List[TravelSubTask]
    is_greeting: bool


os.environ["AZURE_OPENAI_ENDPOINT"] = "https://AZURE_OPENAI_ENDPOINT.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = "AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-2024-08-06"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"




def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


client = AzureOllamaChatCompletionClient(
    azure_deployment=get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model=get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=get_env_variable("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=get_env_variable("AZURE_OPENAI_ENDPOINT"),
#     api_key=get_env_variable("AZURE_OPENAI_API_KEY"),
)

messages = [
    SystemMessage(content="""You are an planner agent.
    Your job is to decide which agents to run based on the user's request.
    Below are the available agents specialised in different tasks:
    - FlightBooking: For booking flights and providing flight information
    - HotelBooking: For booking hotels and providing hotel information
    - CarRental: For booking cars and providing car rental information
    - ActivitiesBooking: For booking activities and providing activity information
    - DestinationInfo: For providing information about destinations
    - DefaultAgent: For handling general requests""", source="system"),
    UserMessage(content="Create a travel plan for a family of 2 kids from Singapore to Melbourne", source="user"),
]


async def run_async_code_bc707437():
    async def run_async_code_b63e0015():
        response = await client.create(messages=messages, extra_create_args={"response_format": TravelPlan})
        return response
    response = asyncio.run(run_async_code_b63e0015())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_bc707437())
logger.success(format_json(response))

response_content: Optional[str] = response.content if isinstance(response.content, str) else None
if response_content is None:
    raise ValueError("Response content is not a valid JSON string")

plogger.debug(json.loads(response_content))

logger.info("\n\n[DONE]", bright=True)