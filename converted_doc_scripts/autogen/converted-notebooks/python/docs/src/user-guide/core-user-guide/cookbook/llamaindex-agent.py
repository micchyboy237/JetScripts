import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLX
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.base.llms.types import (
ChatMessage,
MessageRole,
)
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.embeddings.azure_openai import AzureMLXEmbedding
from llama_index.llms.azure_openai import AzureMLX
from llama_index.tools.wikipedia import WikipediaToolSpec
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using LlamaIndex-Backed Agent

This example demonstrates how to create an AI agent using LlamaIndex.

First install the dependencies:
"""
logger.info("# Using LlamaIndex-Backed Agent")



"""
Let's import the modules.
"""
logger.info("Let's import the modules.")



"""
Define our message type that will be used to communicate with the agent.
"""
logger.info("Define our message type that will be used to communicate with the agent.")

class Resource(BaseModel):
    content: str
    node_id: str
    score: Optional[float] = None


class Message(BaseModel):
    content: str
    sources: Optional[List[Resource]] = None

"""
Define the agent using LLamaIndex's API.
"""
logger.info("Define the agent using LLamaIndex's API.")

class LlamaIndexAgent(RoutedAgent):
    def __init__(self, description: str, llama_index_agent: AgentRunner, memory: BaseMemory | None = None) -> None:
        super().__init__(description)

        self._llama_index_agent = llama_index_agent
        self._memory = memory

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        history_messages: List[ChatMessage] = []

        response: AgentChatResponse  # pyright: ignore
        if self._memory is not None:
            history_messages = self._memory.get(input=message.content)

            async def run_async_code_18a7ce8b():
                response = self._llama_index_agent.chat(message=message.content, history_messages=history_messages)  # pyright: ignore
                return response
            response = asyncio.run(run_async_code_18a7ce8b())
            logger.success(format_json(response))
        else:
            async def run_async_code_27beff32():
                response = self._llama_index_agent.chat(message=message.content)  # pyright: ignore
                return response
            response = asyncio.run(run_async_code_27beff32())
            logger.success(format_json(response))

        if isinstance(response, AgentChatResponse):
            if self._memory is not None:
                self._memory.put(ChatMessage(role=MessageRole.USER, content=message.content))
                self._memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.response))

            assert isinstance(response.response, str)

            resources: List[Resource] = [
                Resource(content=source_node.get_text(), score=source_node.score, node_id=source_node.id_)
                for source_node in response.source_nodes
            ]

            tools: List[Resource] = [
                Resource(content=source.content, node_id=source.tool_name) for source in response.sources
            ]

            resources.extend(tools)
            return Message(content=response.response, sources=resources)
        else:
            return Message(content="I'm sorry, I don't have an answer for you.")

"""
Setting up LlamaIndex.
"""
logger.info("Setting up LlamaIndex.")

llm = MLXAutogenChatLLMAdapter(
    model="qwen3-1.7b-4bit",
    temperature=0.0,
#     api_key=os.getenv("OPENAI_API_KEY"),
)

embed_model = MLXEmbedding(
    model="text-embedding-ada-002",
#     api_key=os.getenv("OPENAI_API_KEY"),
)

Settings.llm = llm
Settings.embed_model = embed_model

"""
Create the tools.
"""
logger.info("Create the tools.")

wiki_spec = WikipediaToolSpec()
wikipedia_tool = wiki_spec.to_tool_list()[1]

"""
Now let's test the agent. First we need to create an agent runtime and
register the agent, by providing the agent's name and a factory function
that will create the agent.
"""
logger.info("Now let's test the agent. First we need to create an agent runtime and")

runtime = SingleThreadedAgentRuntime()
async def async_func_1():
    await LlamaIndexAgent.register(
        runtime,
        "chat_agent",
        lambda: LlamaIndexAgent(
            description="Llama Index Agent",
            llama_index_agent=ReActAgent.from_tools(
                tools=[wikipedia_tool],
                llm=llm,
                max_iterations=8,
                memory=ChatSummaryMemoryBuffer(llm=llm, token_limit=16000),
                verbose=True,
            ),
        ),
    )
asyncio.run(async_func_1())
agent = AgentId("chat_agent", "default")

"""
Start the agent runtime.
"""
logger.info("Start the agent runtime.")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())

"""
Send a direct message to the agent, and print the response.
"""
logger.info("Send a direct message to the agent, and print the response.")

message = Message(content="What are the best movies from studio Ghibli?")
async def run_async_code_97c7fd34():
    response = await runtime.send_message(message, agent)
    return response
response = asyncio.run(run_async_code_97c7fd34())
logger.success(format_json(response))
assert isinstance(response, Message)
logger.debug(response.content)

if response.sources is not None:
    for source in response.sources:
        logger.debug(source.content)

"""
Stop the agent runtime.
"""
logger.info("Stop the agent runtime.")

async def run_async_code_4aaa8dea():
    await runtime.stop()
asyncio.run(run_async_code_4aaa8dea())

logger.info("\n\n[DONE]", bright=True)