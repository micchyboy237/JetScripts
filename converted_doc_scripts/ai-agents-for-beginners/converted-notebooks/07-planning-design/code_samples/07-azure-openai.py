async def main():
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
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    
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
    
    
    response = await client.create(messages=messages, extra_create_args={"response_format": TravelPlan})
    logger.success(format_json(response))
    
    response_content: Optional[str] = response.content if isinstance(response.content, str) else None
    if response_content is None:
        raise ValueError("Response content is not a valid JSON string")
    
    plogger.debug(json.loads(response_content))
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())