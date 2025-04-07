# Example 2: Task - Image Processing (e.g., Resize)
# In a more advanced scenario, let's assume the agent will work with an image processing tool.

import json
import base64
import io
from typing import Optional

from PIL import Image as PILImage

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.logger import logger
from jet.transformers.formatters import format_json


# Simulate an image processing tool that resizes an image
async def resize_image(image_base64: str, width: int, height: int) -> str:
    image_data = base64.b64decode(image_base64)
    image = PILImage.open(io.BytesIO(image_data))
    resized_image = image.resize((width, height))

    # Convert back to base64
    output_buffer = io.BytesIO()
    resized_image.save(output_buffer, format="JPEG")
    resized_image_base64 = base64.b64encode(
        output_buffer.getvalue()).decode("utf-8")

    return resized_image_base64


async def main():
    model_client = OllamaChatCompletionClient(model="gemma3:4b")

    agent = AssistantAgent(
        "image_processing_agent",
        model_client=model_client,
        tools=[FunctionTool(resize_image, description="Resize Image")],
    )

    # Run the agent with an image resizing task
    result = await agent.run(task="resize_image")

    # Output the result (e.g., resized image base64)
    logger.success("Resized Image Base64:", format_json(result))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
