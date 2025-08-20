import asyncio
from jet.transformers.formatters import format_json
from autogen_core.models import UserMessage
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import AzureMLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
from pydantic import BaseModel
from typing import Optional
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Structured output using GPT-4o models

This cookbook demonstrates how to obtain structured output using GPT-4o models. The MLX beta client SDK provides a parse helper that allows you to use your own Pydantic model, eliminating the need to define a JSON schema. This approach is recommended for supported models.

Currently, this feature is supported for:

- llama-3.2-3b-instruct on MLX
- gpt-4o-2024-08-06 on MLX
- gpt-4o-2024-08-06 on Azure

Let's define a simple message type that carries explanation and output for a Math problem
"""
logger.info("# Structured output using GPT-4o models")


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


os.environ["AZURE_OPENAI_ENDPOINT"] = "https://YOUR_ENDPOINT_DETAILS.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-2024-08-06"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"


def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


client = AzureMLXAutogenChatLLMAdapter(
    azure_deployment=get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model=get_env_variable("AZURE_OPENAI_MODEL"),
    api_version=get_env_variable("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=get_env_variable("AZURE_OPENAI_ENDPOINT"),
    #     api_key=get_env_variable("AZURE_OPENAI_API_KEY"),
)

messages = [
    UserMessage(content="What is 16 + 32?", source="user"),
]


async def run_async_code_52c87188():
    async def run_async_code_ca70b240():
        response = await client.create(messages=messages, extra_create_args={"response_format": MathReasoning})
        return response
    response = asyncio.run(run_async_code_ca70b240())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_52c87188())
logger.success(format_json(response))

response_content: Optional[str] = response.content if isinstance(
    response.content, str) else None
if response_content is None:
    raise ValueError("Response content is not a valid JSON string")

logger.debug(json.loads(response_content))

MathReasoning.model_validate(json.loads(response_content))

logger.info("\n\n[DONE]", bright=True)
