import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, HTML
from dotenv import load_dotenv
from jet.logger import CustomLogger
from openai import AsyncOllama
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")






class DestinationsPlugin:
    """A List of Destinations for vacation."""

    @kernel_function(description="Provides a list of vacation destinations.")
    def get_destinations(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Barcelona, Spain
        Paris, France
        Berlin, Germany
        Tokyo, Japan
        New York, USA
        """

    @kernel_function(description="Provides available flight times for a destination.")
    def get_flight_times(
        self, destination: Annotated[str, "The destination to check flight times for."]
    ) -> Annotated[str, "Returns flight times for the specified destination."]:
        return "HTTP ERROR 404: Flight times service is currently unavailable."

    @kernel_function(description="Backup function that provides available flight times for a destination.")
    def get_flight_times_backup(
        self, destination: Annotated[str, "The destination to check flight times for."]
    ) -> Annotated[str, "Returns flight times for the specified destination."]:
        flight_times = {
            "Barcelona": ["08:30 AM", "02:15 PM", "10:45 PM"],
            "Paris": ["06:45 AM", "12:30 PM", "07:15 PM"],
            "Berlin": ["07:20 AM", "01:45 PM", "09:30 PM"],
            "Tokyo": ["11:00 AM", "05:30 PM", "11:55 PM"],
            "New York": ["05:15 AM", "03:00 PM", "08:45 PM"]
        }

        city = destination.split(',')[0].strip()

        if city in flight_times:
            times = ", ".join(flight_times[city])
            return f"Flight times for {city}: {times}"
        else:
            return f"No flight information available for {city}."

load_dotenv()

chat_completion_service = OllamaChatCompletion(
    ai_model_id="llama3.1",
#     api_key=os.getenv("OPENAI_API_KEY"),
)

AGENT_NAME = "TravelAgent"
AGENT_INSTRUCTIONS = """ \
"You are Flight Booking Agent that provides information about available flights and gives travel activity suggestions when asked.
Travel activity suggestions should be specific to customer, location and amount of time at location.

You have access to the following tools to help users plan their trips:
1. get_destinations: Returns a list of available vacation destinations that users can choose from.
2. get_flight_times: Provides available flight times for specific destinations.
3. get_flight_times_backup: Backup function that provides available flight times when the primary service is down.

Your process for assisting users:
- When users inquire about flight booking, book the earliest flight available for the destination they choose using get_flight_times.
- If get_flight_times returns an error message, immediately use get_flight_times_backup with the same destination parameter to retrieve flight information.
- Since you do not have access to a booking system, DO NOT ask to proceed with booking, just assume you have booked the flight.
- Use any past conversation history to understand user preferences and consider them when making suggestions on flights and activities. When making a suggestion, be very clear on why you are making this suggestion if based on a user preference.

Guidelines:
- Use the exact destination names when using tools (Barcelona, Paris, Berlin, Tokyo, New York)
- Respond in a helpful and enthusiastic manner about travel possibilities
- Always seek feedback to ensure your suggestions meet the user's expectations
- Acknowledge when a request falls outside your capabilities
- For better formatting, always display flight times in a list format
- When giving any timed suggestions, reflect if the time frames are reasonable. Respond again if not.
- If the flight times service is down, inform the user that you're using backup flight data while maintaining a positive tone.

Your goal is to help users explore vacation options efficiently and make informed travel decisions by understanding their preferences and providing tailored recommendations.
"""
agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin()],
    name=AGENT_NAME,
    instructions=AGENT_INSTRUCTIONS,
)


user_inputs = [
    "Book me a flight to Barcelona",
]

thread: ChatHistoryAgentThread | None = None

async def main():
    global thread

    for user_input in user_inputs:
        html_output = (
            f"<div style='margin-bottom:10px'>"
            f"<div style='font-weight:bold'>User:</div>"
            f"<div style='margin-left:20px'>{user_input}</div></div>"
        )

        agent_name = None
        full_response: list[str] = []
        function_calls: list[str] = []

        current_function_name = None
        argument_buffer = ""

        async for response in agent.invoke_stream(
            messages=user_input,
            thread=thread,
        ):
            thread = response.thread
            agent_name = response.name
            content_items = list(response.items)

            for item in content_items:
                if isinstance(item, FunctionCallContent):
                    if item.function_name:
                        current_function_name = item.function_name

                    if isinstance(item.arguments, str):
                        argument_buffer += item.arguments
                elif isinstance(item, FunctionResultContent):
                    if current_function_name:
                        formatted_args = argument_buffer.strip()
                        try:
                            parsed_args = json.loads(formatted_args)
                            formatted_args = json.dumps(parsed_args)
                        except Exception:
                            pass  # leave as raw string

                        function_calls.append(f"Calling function: {current_function_name}({formatted_args})")
                        current_function_name = None
                        argument_buffer = ""

                    function_calls.append(f"\nFunction Result:\n\n{item.result}")
                elif isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)

        if function_calls:
            html_output += (
                "<div style='margin-bottom:10px'>"
                "<details>"
                "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>Function Calls (click to expand)</summary>"
                "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
                "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap; font-size:14px; color:#333;'>"
                f"{chr(10).join(function_calls)}"
                "</div></details></div>"
            )

        html_output += (
            "<div style='margin-bottom:20px'>"
            f"<div style='font-weight:bold'>{agent_name or 'Assistant'}:</div>"
            f"<div style='margin-left:20px; white-space:pre-wrap'>{''.join(full_response)}</div></div><hr>"
        )

        display(HTML(html_output))

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)