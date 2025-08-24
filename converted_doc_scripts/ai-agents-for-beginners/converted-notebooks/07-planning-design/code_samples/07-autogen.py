import asyncio
from jet.transformers.formatters import format_json
from autogen_core.models import UserMessage, SystemMessage, AssistantMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from enum import Enum
from jet.logger import CustomLogger
from pprint import pprint
from pydantic import BaseModel
from typing import List, Optional, Union
from typing import Optional
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


load_dotenv()


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
    assigned_agent: AgentEnum  # we want to assign the task to the agent


class TravelPlan(BaseModel):
    main_task: str
    subtasks: List[TravelSubTask]
    is_greeting: bool


client = AzureAIChatCompletionClient(
    model="llama3.1",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

messages = [
    SystemMessage(content="""You are an planner agent.
    Your job is to decide which agents to run based on the user's request.
                      Provide your response in JSON format with the following structure:
{'main_task': 'Plan a family trip from Singapore to Melbourne.',
 'subtasks': [{'assigned_agent': 'flight_booking',
               'task_details': 'Book round-trip flights from Singapore to '
                               'Melbourne.'}
    Below are the available agents specialised in different tasks:
    - FlightBooking: For booking flights and providing flight information
    - HotelBooking: For booking hotels and providing hotel information
    - CarRental: For booking cars and providing car rental information
    - ActivitiesBooking: For booking activities and providing activity information
    - DestinationInfo: For providing information about destinations
    - DefaultAgent: For handling general requests""", source="system"),
    UserMessage(
        content="Create a travel plan for a family of 2 kids from Singapore to Melbourne", source="user"),
]


async def run_async_code_51d41b79():
    async def run_async_code_f4199a31():
        response = await client.create(messages=messages, extra_create_args={"response_format": 'json_object'})
        return response
    response = asyncio.run(run_async_code_f4199a31())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_51d41b79())
logger.success(format_json(response))


response_content: Optional[str] = response.content if isinstance(
    response.content, str) else None
if response_content is None:
    raise ValueError("Response content is not a valid JSON string")

plogger.debug(json.loads(response_content))

logger.info("\n\n[DONE]", bright=True)
