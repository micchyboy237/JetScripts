import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from pydantic import BaseModel
from typing import Optional
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Controlling Agent Reasoning Loop with Return Direct Tools

All tools have an option for `return_direct` -- if this is set to `True`, and the associated tool is called (without any other tools being called), the agent reasoning loop is ended and the tool output is returned directly.

This can be useful for speeding up response times when you know the tool output is good enough, to avoid the agent re-writing the response, and for ending the reasoning loop.

This notebook walks through a notebook where an agent needs to gather information from a user in order to make a restaurant booking.
"""
logger.info("# Controlling Agent Reasoning Loop with Return Direct Tools")

# %pip install llama-index-core llama-index-llms-anthropic


# os.environ["ANTHROPIC_API_KEY"] = "sk-..."

"""
## Tools setup
"""
logger.info("## Tools setup")



bookings = {}


class Booking(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


def get_booking_state(user_id: str) -> str:
    """Get the current state of a booking for a given booking ID."""
    try:
        return str(bookings[user_id].dict())
    except:
        return f"Booking ID {user_id} not found"


def update_booking(user_id: str, property: str, value: str) -> str:
    """Update a property of a booking for a given booking ID. Only enter details that are explicitly provided."""
    booking = bookings[user_id]
    setattr(booking, property, value)
    return f"Booking ID {user_id} updated with {property} = {value}"


def create_booking(user_id: str) -> str:
    """Create a new booking and return the booking ID."""
    bookings[user_id] = Booking()
    return "Booking created, but not yet confirmed. Please provide your name, email, phone, date, and time."


def confirm_booking(user_id: str) -> str:
    """Confirm a booking for a given booking ID."""
    booking = bookings[user_id]

    if booking.name is None:
        raise ValueError("Please provide your name.")

    if booking.email is None:
        raise ValueError("Please provide your email.")

    if booking.phone is None:
        raise ValueError("Please provide your phone number.")

    if booking.date is None:
        raise ValueError("Please provide the date of your booking.")

    if booking.time is None:
        raise ValueError("Please provide the time of your booking.")

    return f"Booking ID {user_id} confirmed!"


get_booking_state_tool = FunctionTool.from_defaults(fn=get_booking_state)
update_booking_tool = FunctionTool.from_defaults(fn=update_booking)
create_booking_tool = FunctionTool.from_defaults(
    fn=create_booking, return_direct=True
)
confirm_booking_tool = FunctionTool.from_defaults(
    fn=confirm_booking, return_direct=True
)

"""
## A user has walked in! Let's help them make a booking
"""
logger.info("## A user has walked in! Let's help them make a booking")


llm = Anthropic(model="claude-3-sonnet-20240229", temperature=0.1)

user = "user123"
system_prompt = f"""You are now connected to the booking system and helping {user} with making a booking.
Only enter details that the user has explicitly provided.
Do not make up any details.
"""

agent = FunctionAgent(
    tools=[
        get_booking_state_tool,
        update_booking_tool,
        create_booking_tool,
        confirm_booking_tool,
    ],
    llm=llm,
    system_prompt=system_prompt,
)

ctx = Context(agent)


handler = agent.run(
    "Hello! I would like to make a booking, around 5pm?", ctx=ctx
)

async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        logger.debug(f"{ev.delta}", end="", flush=True)
    elif isinstance(ev, ToolCallResult):
        logger.debug(
            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}"
        )

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.debug(str(response))

"""
Perfect, we can see the function output was retruned directly, with no modification or final LLM call!
"""
logger.info("Perfect, we can see the function output was retruned directly, with no modification or final LLM call!")

handler = agent.run(
    "Sure! My name is Logan, and my email is test@gmail.com?", ctx=ctx
)

async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        logger.debug(f"{ev.delta}", end="", flush=True)
    elif isinstance(ev, ToolCallResult):
        logger.debug(
            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}"
        )

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.debug(str(response))

handler = agent.run(
    "Right! My phone number is 1234567890, the date of the booking is April 5, at 5pm.",
    ctx=ctx,
)

async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        logger.debug(f"{ev.delta}", end="", flush=True)
    elif isinstance(ev, ToolCallResult):
        logger.debug(
            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}"
        )

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.debug(str(response))

logger.debug(bookings["user123"])

logger.info("\n\n[DONE]", bright=True)