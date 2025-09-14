# Example 1: Task - Query User Data from a Database
# This scenario will simulate querying a database for user details, and then using the AssistantAgent to handle the task.

import json
import logging

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import AssistantAgent

from autogen_core import FunctionCall
from autogen_core.models import (
    CreateResult,
    RequestUsage,
)
from autogen_core.models._model_client import ModelFamily
from autogen_core.tools import FunctionTool
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json


# Simulate a database query tool
async def query_user_data(user_id: str) -> str:
    # In real-world, this would involve querying a database
    user_data = {
        "123": {"name": "John Doe", "email": "john.doe@example.com"},
        "456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
    }
    return json.dumps(user_data.get(user_id, {"error": "User not found"}))


# Function for greeting the user based on query result
async def greet_user(user_id: str) -> str:
    user_data = await query_user_data(user_id)
    return f"Hello, {json.loads(user_data).get('name', 'Guest')}!"


async def main():
    model_client = OllamaChatCompletionClient(model="llama3.2")

    agent = AssistantAgent(
        "user_data_agent",
        model_client=model_client,
        tools=[FunctionTool(query_user_data, description="Query User Data"), FunctionTool(
            greet_user, description="Greet User")],
    )

    # Run the agent with the user_id task
    result = await agent.run(task="123")

    # Output the result
    logger.success("Result messages:", format_json(result))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
