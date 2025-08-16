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
client = AsyncOllama(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

chat_completion_service = OllamaChatCompletion(
    ai_model_id="llama3.1",
    async_client=client,
)

AGENT_NAME = "TravelAgent"
AGENT_INSTRUCTIONS = """ \
"You are Flight Booking Agent that provides information about available flights and gives travel activity suggestions when asked.
Travel activity suggestions should be specific to customer, location and amount of time at location.

You have access to the following tools to help users plan their trips:
1. get_destinations: Returns a list of available vacation destinations that users can choose from.
2. get_flight_times: Provides available flight times for specific destinations.


Your process for assisting users:
- When users first inquire about flight booking with no prior history, ask for their preferred flight time ONCE.
- MAINTAIN a customer_preferences object throughout the conversation to track preferred flight times.
- When a user books a flight to any destination, RECORD their chosen flight time in the customer_preferences object.
- For ALL subsequent flight inquiries to ANY destination, AUTOMATICALLY apply their existing preferred flight time without asking.
- NEVER ask about time preferences again after they've been established for any destination.
- When suggesting flights for a new destination, explicitly say: "Based on your previous preference for [time] flights, I recommend..."
- Only after showing options matching their preferred time, ask if they want to see alternative times.
- After each booking, UPDATE the customer_preferences object with any new information.
- ALWAYS mention which specific preference you used when making a suggestion.

Guidelines:
- Use the exact destination names when using tools (Barcelona, Paris, Berlin, Tokyo, New York)
- Respond in a helpful and enthusiastic manner about travel possibilities
- Always seek feedback to ensure your suggestions meet the user's expectations
- Acknowledge when a request falls outside your capabilities
- For better formatting, always display flight times in a list format
- When giving any timed suggestions, reflect if the time frames are reasonable. Respond again if not.

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
    "I prefer a later flight",
    "That is too late, choose the earliest flight",
    "I want to leave the same day, give me some suggestions of things to do in Barcelona during my layover if I take the last flight out",
    "I am stressed this wont be enough time"
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

async def continue_chat():
    global thread

    user_inputs = [
        "Book me a flight to Paris",
    ]

    for user_input in user_inputs:
        html_output = "<div style='margin-bottom:10px'>"
        html_output += "<div style='font-weight:bold'>User:</div>"
        html_output += f"<div style='margin-left:20px'>{user_input}</div>"
        html_output += "</div>"

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

async def run_async_code_dd2c1577():
    await continue_chat()
    return 
 = asyncio.run(run_async_code_dd2c1577())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)