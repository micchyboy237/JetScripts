async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, HTML
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from openai import AsyncOllama
    from pydantic import BaseModel, ValidationError, Field
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
    from semantic_kernel.functions import KernelArguments
    from typing import List
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
    
    
    
    
    
    
    
    load_dotenv()
    client = AsyncOllama(
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com/",
    )
    
    chat_completion_service = OllamaChatCompletion(
        ai_model_id="llama3.2",
        async_client=client,
    )
    
    class SubTask(BaseModel):
        assigned_agent: str = Field(
            description="The specific agent assigned to handle this subtask")
        task_details: str = Field(
            description="Detailed description of what needs to be done for this subtask")
    
    
    class TravelPlan(BaseModel):
        main_task: str = Field(
            description="The overall travel request from the user")
        subtasks: List[SubTask] = Field(
            description="List of subtasks broken down from the main task, each assigned to a specialized agent")
    
    AGENT_NAME = "TravelAgent"
    AGENT_INSTRUCTIONS = """You are an planner agent.
        Your job is to decide which agents to run based on the user's request.
        Below are the available agents specialised in different tasks:
        - FlightBooking: For booking flights and providing flight information
        - HotelBooking: For booking hotels and providing hotel information
        - CarRental: For booking cars and providing car rental information
        - ActivitiesBooking: For booking activities and providing activity information
        - DestinationInfo: For providing information about destinations
        - DefaultAgent: For handling general requests"""
    
    settings = OllamaChatPromptExecutionSettings(response_format=TravelPlan)
    
    agent = ChatCompletionAgent(
        service=chat_completion_service,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(settings)
    )
    
    
    
    async def main():
        thread: ChatHistoryAgentThread | None = None
    
        user_inputs = [
            "Create a travel plan for a family of 4, with 2 kids, from Singapore to Melbourne",
        ]
    
        for user_input in user_inputs:
    
            html_output = "<div style='margin-bottom:10px'>"
            html_output += "<div style='font-weight:bold'>User:</div>"
            html_output += f"<div style='margin-left:20px'>{user_input}</div>"
            html_output += "</div>"
    
            response = await agent.get_response(messages=user_input, thread=thread)
            logger.success(format_json(response))
            thread = response.thread
    
            try:
                travel_plan = TravelPlan.model_validate(json.loads(response.message.content))
    
                formatted_json = travel_plan.model_dump_json(indent=4)
                html_output += "<div style='margin-bottom:20px'>"
                html_output += "<div style='font-weight:bold'>Validated Travel Plan:</div>"
                html_output += f"<pre style='margin-left:20px; padding:10px; border-radius:5px;'>{formatted_json}</pre>"
                html_output += "</div>"
            except ValidationError as e:
                html_output += "<div style='margin-bottom:20px; color:red;'>"
                html_output += "<div style='font-weight:bold'>Validation Error:</div>"
                html_output += f"<pre style='margin-left:20px;'>{str(e)}</pre>"
                html_output += "</div>"
                html_output += "<div style='margin-bottom:20px;'>"
                html_output += "<div style='font-weight:bold'>Raw Response:</div>"
                html_output += f"<div style='margin-left:20px; white-space:pre-wrap'>{response.content}</div>"
                html_output += "</div>"
    
            html_output += "<hr>"
    
            display(HTML(html_output))
    
    await main()
    
    """
    You should see sample output similar to:
    
    ```json
    User:
    Create a travel plan for a family of 4, with 2 kids, from Singapore to Melboune
    Validated Travel Plan:
    {
        "main_task": "Plan a family trip from Singapore to Melbourne for 4 people including 2 kids.",
        "subtasks": [
            {
                "assigned_agent": "FlightBooking",
                "task_details": "Book round-trip flights from Singapore to Melbourne for 2 adults and 2 children."
            },
            {
                "assigned_agent": "HotelBooking",
                "task_details": "Find and book a family-friendly hotel in Melbourne that accommodates 4 people."
            },
            {
                "assigned_agent": "CarRental",
                "task_details": "Arrange for a car rental in Melbourne suitable for a family of 4."
            },
            {
                "assigned_agent": "ActivitiesBooking",
                "task_details": "Plan and book family-friendly activities in Melbourne suitable for kids."
            },
            {
                "assigned_agent": "DestinationInfo",
                "task_details": "Provide information about Melbourne, including attractions, dining options, and family-oriented activities."
            }
        ]
    }
    ```
    """
    logger.info("You should see sample output similar to:")
    
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