import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Quickstart

Via AgentChat, you can build applications quickly using preset agents.
To illustrate this, we will begin with creating a single agent that can
use tools.

First, we need to install the AgentChat and Extension packages.
"""
logger.info("# Quickstart")

pip install - U "autogen-agentchat" "autogen-ext[openai,azure]"

"""
This example uses an MLX model, however, you can use other models as well.
Simply update the `model_client` with the desired model or model client class.

To use Azure MLX models and AAD authentication,
you can follow the instructions [here](./tutorial/models.ipynb#azure-openai).
To use other models, see [Models](./tutorial/models.ipynb).
"""
logger.info(
    "This example uses an MLX model, however, you can use other models as well.")


model_client = MLXAutogenChatLLMAdapter(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)


async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


async def main() -> None:
    async def run_async_code_654bf0df():
        await Console(agent.run_stream(task="What is the weather in New York?"))
        return
     = asyncio.run(run_async_code_654bf0df())
    logger.success(format_json())
    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())


async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

"""
## What's Next?

Now that you have a basic understanding of how to use a single agent, consider following the [tutorial](./tutorial/index.md) for a walkthrough on other features of AgentChat.
"""
logger.info("## What's Next?")

logger.info("\n\n[DONE]", bright=True)