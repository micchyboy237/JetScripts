import asyncio
from jet.transformers.formatters import format_json
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from jet.logger import CustomLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent Chat with Async Human Inputs
"""
logger.info("# Agent Chat with Async Human Inputs")

# %pip install "autogen" chromadb sentence_transformers tiktoken pypdf nest-asyncio


# import nest_asyncio


async def my_asynchronous_function():
    logger.debug("Start asynchronous function")
    async def run_async_code_6d53c47d():
        await asyncio.sleep(2)  # Simulate some asynchronous task (e.g., I/O operation)
        return 
     = asyncio.run(run_async_code_6d53c47d())
    logger.success(format_json())
    logger.debug("End asynchronous function")
    return "input"




class CustomisedUserProxyAgent(UserProxyAgent):
    async def a_get_human_input(self, prompt: str) -> str:
        async def run_async_code_2c3867b0():
            async def run_async_code_53d53fc1():
                user_input = await my_asynchronous_function()
                return user_input
            user_input = asyncio.run(run_async_code_53d53fc1())
            logger.success(format_json(user_input))
            return user_input
        user_input = asyncio.run(run_async_code_2c3867b0())
        logger.success(format_json(user_input))

        return user_input


    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        async def run_async_code_99c5ee1f():
            await super().a_receive(message, sender, request_reply, silent)
            return 
         = asyncio.run(run_async_code_99c5ee1f())
        logger.success(format_json())


class CustomisedAssistantAgent(AssistantAgent):
    async def a_get_human_input(self, prompt: str) -> str:
        async def run_async_code_2c3867b0():
            async def run_async_code_53d53fc1():
                user_input = await my_asynchronous_function()
                return user_input
            user_input = asyncio.run(run_async_code_53d53fc1())
            logger.success(format_json(user_input))
            return user_input
        user_input = asyncio.run(run_async_code_2c3867b0())
        logger.success(format_json(user_input))

        return user_input

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        async def run_async_code_99c5ee1f():
            await super().a_receive(message, sender, request_reply, silent)
            return 
         = asyncio.run(run_async_code_99c5ee1f())
        logger.success(format_json())

def create_llm_config(model, temperature, seed):
    config_list = [
        {
            "model": "<model_name>",
            "api_key": "<api_key>",
        },
    ]

    llm_config = {
        "seed": int(seed),
        "config_list": config_list,
        "temperature": float(temperature),
    }

    return llm_config

# nest_asyncio.apply()


async def main():
    boss = CustomisedUserProxyAgent(
        name="boss",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = CustomisedAssistantAgent(
        name="assistant",
        system_message="You will provide some agenda, and I will create questions for an interview meeting. Every time when you generate question then you have to ask user for feedback and if user provides the feedback then you have to incorporate that feedback and generate new set of questions and if user don't want to update then terminate the process and exit",
        llm_config=create_llm_config("gpt-4", "0.4", "23"),
    )

    await boss.a_initiate_chat(
        assistant,
        message="Resume Review, Technical Skills Assessment, Project Discussion, Job Role Expectations, Closing Remarks.",
        n_results=3,
    )


async def run_async_code_6bc25f65():
    await main()  # noqa: F704
    return 
 = asyncio.run(run_async_code_6bc25f65())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)