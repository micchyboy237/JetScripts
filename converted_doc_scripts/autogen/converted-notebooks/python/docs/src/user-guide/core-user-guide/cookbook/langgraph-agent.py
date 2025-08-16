import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dataclasses import dataclass
from jet.llm.ollama.base_langchain import AzureChatOllama, ChatOllama
from jet.logger import CustomLogger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool  # pyright: ignore
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Any, Callable, List, Literal
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using LangGraph-Backed Agent

This example demonstrates how to create an AI agent using LangGraph.
Based on the example in the LangGraph documentation:
https://langchain-ai.github.io/langgraph/.

First install the dependencies:
"""
logger.info("# Using LangGraph-Backed Agent")



"""
Let's import the modules.
"""
logger.info("Let's import the modules.")



"""
Define our message type that will be used to communicate with the agent.
"""
logger.info("Define our message type that will be used to communicate with the agent.")

@dataclass
class Message:
    content: str

"""
Define the tools the agent will use.
"""
logger.info("Define the tools the agent will use.")

@tool  # pyright: ignore
def get_weather(location: str) -> str:
    """Call to surf the web."""
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

"""
Define the agent using LangGraph's API.
"""
logger.info("Define the agent using LangGraph's API.")

class LangGraphToolUseAgent(RoutedAgent):
    def __init__(self, description: str, model: ChatOllama, tools: List[Callable[..., Any]]) -> None:  # pyright: ignore
        super().__init__(description)
        self._model = model.bind_tools(tools)  # pyright: ignore

        def should_continue(state: MessagesState) -> Literal["tools", END]:  # type: ignore
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:  # type: ignore
                return "tools"
            return END

        async def call_model(state: MessagesState):  # type: ignore
            messages = state["messages"]
            async def run_async_code_984bb2f3():
                async def run_async_code_74d3f8e8():
                    response = await self._model.ainvoke(messages)
                    return response
                response = asyncio.run(run_async_code_74d3f8e8())
                logger.success(format_json(response))
                return response
            response = asyncio.run(run_async_code_984bb2f3())
            logger.success(format_json(response))
            return {"messages": [response]}

        tool_node = ToolNode(tools)  # pyright: ignore

        self._workflow = StateGraph(MessagesState)

        self._workflow.add_node("agent", call_model)  # pyright: ignore
        self._workflow.add_node("tools", tool_node)  # pyright: ignore

        self._workflow.set_entry_point("agent")

        self._workflow.add_conditional_edges(
            "agent",
            should_continue,  # type: ignore
        )

        self._workflow.add_edge("tools", "agent")

        self._app = self._workflow.compile()

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        async def async_func_37():
            final_state = await self._app.ainvoke(
                {
                    "messages": [
                        SystemMessage(
                            content="You are a helpful AI assistant. You can use tools to help answer questions."
                        ),
                        HumanMessage(content=message.content),
                    ]
                },
                config={"configurable": {"thread_id": 42}},
            )
            return final_state
        final_state = asyncio.run(async_func_37())
        logger.success(format_json(final_state))
        response = Message(content=final_state["messages"][-1].content)
        return response

"""
Now let's test the agent. First we need to create an agent runtime and
register the agent, by providing the agent's name and a factory function
that will create the agent.
"""
logger.info("Now let's test the agent. First we need to create an agent runtime and")

runtime = SingleThreadedAgentRuntime()
await LangGraphToolUseAgent.register(
    runtime,
    "langgraph_tool_use_agent",
    lambda: LangGraphToolUseAgent(
        "Tool use agent",
        ChatOllama(
            model="llama3.1", request_timeout=300.0, context_window=4096,
        ),
        [get_weather],
    ),
)
agent = AgentId("langgraph_tool_use_agent", key="default")

"""
Start the agent runtime.
"""
logger.info("Start the agent runtime.")

runtime.start()

"""
Send a direct message to the agent, and print the response.
"""
logger.info("Send a direct message to the agent, and print the response.")

async def run_async_code_272b50c8():
    async def run_async_code_a4af7f9f():
        response = await runtime.send_message(Message("What's the weather in SF?"), agent)
        return response
    response = asyncio.run(run_async_code_a4af7f9f())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_272b50c8())
logger.success(format_json(response))
logger.debug(response.content)

"""
Stop the agent runtime.
"""
logger.info("Stop the agent runtime.")

async def run_async_code_4aaa8dea():
    await runtime.stop()
    return 
 = asyncio.run(run_async_code_4aaa8dea())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)