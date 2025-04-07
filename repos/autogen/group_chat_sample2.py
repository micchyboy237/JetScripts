import asyncio
import logging
from typing import Any, Mapping, Sequence
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_core import AgentRuntime, CancellationToken
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    StopMessage,
    TextMessage,
)
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.agents import (
    BaseChatAgent,
)
from autogen_core.models import (
    UserMessage,
)


# Configure logger
logger = logging.getLogger("real_world_example")
logger.setLevel(logging.DEBUG)


def _pass_function(input: str) -> str:
    return "pass"


class _EchoAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._last_message: str | None = None
        self._total_messages = 0

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    @property
    def total_messages(self) -> int:
        return self._total_messages

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if len(messages) > 0:
            assert isinstance(messages[0], TextMessage)
            self._last_message = messages[0].content
            self._total_messages += 1
            return Response(chat_message=TextMessage(content=messages[0].content, source=self.name))
        else:
            assert self._last_message is not None
            self._total_messages += 1
            return Response(chat_message=TextMessage(content=self._last_message, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_message = None


class _FlakyAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._last_message: str | None = None
        self._total_messages = 0

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    @property
    def total_messages(self) -> int:
        return self._total_messages

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        raise ValueError("I am a flaky agent...")

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_message = None


class _UnknownMessageType(BaseChatMessage):
    content: str

    def to_model_message(self) -> UserMessage:
        raise NotImplementedError("This message type is not supported.")

    def to_model_text(self) -> str:
        raise NotImplementedError("This message type is not supported.")

    def to_text(self) -> str:
        raise NotImplementedError("This message type is not supported.")

    def dump(self) -> Mapping[str, Any]:
        return {}

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> "_UnknownMessageType":
        return cls(**data)


class RealWorldExample:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    async def run_round_robin_chat(self):
        # Simulate a team of agents running in a round-robin fashion
        agent_1 = _EchoAgent("agent_1", description="echo agent 1")
        agent_2 = _FlakyAgent("agent_2", description="flaky agent 2")
        agent_3 = _EchoAgent("agent_3", description="echo agent 3")

        team = RoundRobinGroupChat(
            participants=[agent_1, agent_2, agent_3],
            max_turns=5,
            runtime=self.runtime
        )
        task = "Write a program that prints 'Hello, world!'"
        result = await team.run(task=task)

        logger.info(
            f"Round-robin group chat finished with {len(result.messages)} messages.")
        for message in result.messages:
            logger.info(f"Message from {message.source}: {message.content}")

    async def run_selector_chat(self):
        # Simulate a team of agents using selector logic
        model_client = ReplayChatCompletionClient(
            ["agent1", "agent2", "agent3"])
        agent1 = _EchoAgent("agent1", description="echo agent 1")
        agent2 = _EchoAgent("agent2", description="echo agent 2")
        agent3 = _EchoAgent("agent3", description="echo agent 3")

        team = SelectorGroupChat(
            participants=[agent1, agent2, agent3],
            model_client=model_client,
            max_turns=5,
            runtime=self.runtime
        )
        task = "Write a program that prints 'Hello, world!'"
        result = await team.run(task=task)

        logger.info(
            f"Selector group chat finished with {len(result.messages)} messages.")
        for message in result.messages:
            logger.info(f"Message from {message.source}: {message.content}")

    async def run_task_with_function_tool(self):
        # Execute a task using FunctionTool
        tool = FunctionTool(_pass_function)
        result = await tool.run("Some input to the function tool")

        logger.info(f"Function tool execution result: {result}")

    async def run_chat_with_cancellation(self):
        # Simulate cancellation of a chat session
        agent_1 = _EchoAgent("agent_1", description="echo agent 1")
        agent_2 = _FlakyAgent("agent_2", description="flaky agent 2")
        agent_3 = _EchoAgent("agent_3", description="echo agent 3")

        team = RoundRobinGroupChat(
            participants=[agent_1, agent_2, agent_3],
            max_turns=10,
            runtime=self.runtime
        )
        cancellation_token = CancellationToken()
        run_task = asyncio.create_task(team.run(
            task="Write a program that prints 'Hello, world!'", cancellation_token=cancellation_token))

        await asyncio.sleep(0.1)
        cancellation_token.cancel()

        try:
            await run_task
        except asyncio.CancelledError:
            logger.info("Task was cancelled successfully.")

        # Resume after cancellation
        result = await team.run(task="Write a program that prints 'Hello, world!'")
        logger.info(f"Chat resumed with {len(result.messages)} messages.")
        for message in result.messages:
            logger.info(f"Message from {message.source}: {message.content}")


async def main():
    # Start the agent runtime (could be single-threaded or embedded)
    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    # Initialize the real-world examples
    example = RealWorldExample(runtime=runtime)

    # Run the examples
    await example.run_round_robin_chat()
    await example.run_selector_chat()
    await example.run_task_with_function_tool()
    await example.run_chat_with_cancellation()

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
