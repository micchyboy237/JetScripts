import asyncio
import logging
import json
from typing import Any, List, Mapping, Sequence

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response, TaskResult, TerminationCondition
from autogen_agentchat.messages import (
    BaseChatMessage,
    HandoffMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_agentchat.teams import Swarm
from autogen_core import AgentId, AgentRuntime, CancellationToken, FunctionCall
from autogen_ext.models.replay import ReplayChatCompletionClient
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)


class _EchoAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._last_message = None
        self._total_messages = 0

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    @property
    def total_messages(self) -> int:
        return self._total_messages

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if len(messages) > 0:
            self._last_message = messages[0].content
            self._total_messages += 1
            return Response(chat_message=TextMessage(content=messages[0].content, source=self.name))
        else:
            return Response(chat_message=TextMessage(content=self._last_message, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_message = None


class _HandOffAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, next_agent: str) -> None:
        super().__init__(name, description)
        self._next_agent = next_agent

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (HandoffMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        return Response(
            chat_message=HandoffMessage(
                content=f"Transferred to {self._next_agent}.", target=self._next_agent, source=self.name)
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class _ErrorHandlingAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._total_messages = 0

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        try:
            if len(messages) > 0:
                # Simulate a failure in processing
                if self._total_messages == 2:
                    raise ValueError("Something went wrong!")
                self._total_messages += 1
                return Response(chat_message=TextMessage(content=messages[0].content, source=self.name))
            else:
                return Response(chat_message=TextMessage(content="No message received.", source=self.name))
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {e}")
            return Response(chat_message=TextMessage(content="An error occurred.", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._total_messages = 0


class _SwarmExampleTask(BaseModel):
    task: str
    data: List[str]


async def run_example_task() -> None:
    # Example: Task with HandOff agents
    first_agent = _HandOffAgent(
        "first_agent", description="First Agent", next_agent="second_agent")
    second_agent = _HandOffAgent(
        "second_agent", description="Second Agent", next_agent="third_agent")
    third_agent = _HandOffAgent(
        "third_agent", description="Third Agent", next_agent="first_agent")

    termination = TerminationCondition()  # Example termination condition
    team = Swarm([first_agent, second_agent, third_agent],
                 termination_condition=termination)

    result = await team.run(task="Handle task with agent handoff")
    logger.info(f"Swarm Result: {result}")

    # Simulate tool call handling with error
    model_client = ReplayChatCompletionClient(
        chat_completions=[
            "Hello",
            "TERMINATE"
        ],
        model_info={"family": "gpt-4o", "function_calling": True},
    )
    agent = AssistantAgent("tool_call_agent", model_client=model_client)
    team_with_tool = Swarm([agent], termination_condition=termination)

    result_with_tool = await team_with_tool.run(task="Task involving tool calls")
    logger.info(f"Swarm Result with Tool Calls: {result_with_tool}")

    # Simulating error handling
    error_agent = _ErrorHandlingAgent(
        "error_agent", description="Agent with error handling")
    team_with_error_handling = Swarm(
        [error_agent], termination_condition=termination)

    result_with_error = await team_with_error_handling.run(task="Task with error handling")
    logger.info(f"Swarm Result with Error Handling: {result_with_error}")


async def main() -> None:
    await run_example_task()

# Run the example task
if __name__ == "__main__":
    asyncio.run(main())
