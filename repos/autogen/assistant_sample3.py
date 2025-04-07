# Example 3: Task - Perform a Complex Calculation
# In another example, the agent could be tasked with performing a complex calculation, such as computing the area of a circle.

import json
import math

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json


# Function to calculate the area of a circle
async def calculate_area_of_circle(radius: float) -> str:
    area = math.pi * radius * radius
    return f"The area of the circle is {area:.2f}"


async def main():
    model_client = OllamaChatCompletionClient(model="llama3.2")

    agent = AssistantAgent(
        "calculation_agent",
        model_client=model_client,
        tools=[FunctionTool(calculate_area_of_circle,
                            description="Calculate Area of Circle")],
    )

    # Run the agent with the circle area calculation task
    result = await agent.run(task="calculate_area_of_circle")

    # Output the result
    logger.success("Calculation result::", format_json(result))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
