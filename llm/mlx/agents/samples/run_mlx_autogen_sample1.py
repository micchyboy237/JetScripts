# Example 1: Task - Query User Data from a Database
# This scenario will simulate querying a database for user details, and then using the AssistantAgent to handle the task.

import json
import os
import shutil

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import AssistantAgent

from autogen_core import FunctionCall
from autogen_core.models import (
    CreateResult,
    RequestUsage,
)
from autogen_core.models._model_client import ModelFamily
from autogen_core.tools import FunctionTool
from jet.file.utils import save_file
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Simulate a database query tool


async def query_user_data(user_id: str) -> str:
    # In real-world, this would involve querying a database
    logger.info(f"Called query_user_data. Params: (user_id={user_id})")
    user_data = {
        "123": {"name": "John Doe", "email": "john.doe@example.com"},
        "456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
    }
    return json.dumps(user_data.get(user_id, {"error": "User not found"}))


# Function for greeting the user based on query result
async def greet_user(user_id: str) -> str:
    logger.info(f"Called greet_user. Params: (user_id={user_id})")
    user_data = await query_user_data(user_id)
    return f"Hello, {json.loads(user_data).get('name', 'Guest')}!"


async def main():
    model_client = MLXChatCompletionClient(
        model="llama-3.2-3b-instruct-4bit", log_dir=f"{OUTPUT_DIR}/chats")
    query_user_data_tool = FunctionTool(
        query_user_data, description="Query User Data")
    greet_user_tool = FunctionTool(greet_user, description="Greet User")

    agent = AssistantAgent(
        "user_data_agent",
        model_client=model_client,
        tools=[
            query_user_data_tool,
            greet_user_tool
        ],
        reflect_on_tool_use=True,
    )

    # Run the agent with the user_id task
    result = await agent.run(task="123")

    # Output the result
    logger.success("Result messages:", format_json(result))
    save_file(result, f"{OUTPUT_DIR}/task_result.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
