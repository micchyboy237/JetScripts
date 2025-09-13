import asyncio
import json
import logging
from typing import Any, AsyncGenerator, List, Mapping, Sequence

# Libraries simulating Agent system (mocked here)
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent, CodeExecutorAgent
from autogen_agentchat.base import Handoff, Response, TaskResult, TerminationCondition
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import AgentRuntime, CancellationToken, SingleThreadedAgentRuntime
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    StructuredMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.conditions import MaxMessageTermination

logger = logging.getLogger("example_logger")
logger.setLevel(logging.DEBUG)


class EchoAgent(BaseChatAgent):
    """A real-world example where the agent responds with echo messages."""

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
            self._last_message = messages[0].content
            self._total_messages += 1
            return Response(chat_message=TextMessage(content=messages[0].content, source=self.name))
        else:
            return Response(chat_message=TextMessage(content=self._last_message, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_message = None


class CustomerSupportAgent(EchoAgent):
    """A real-world customer support agent handling common inquiries."""

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if "issue" in messages[0].content.lower():
            return Response(chat_message=TextMessage(content="Sorry to hear about your issue. Let me assist you.", source=self.name))
        return await super().on_messages(messages, cancellation_token)


class HelpDeskAgent(EchoAgent):
    """A Help Desk agent offering troubleshooting steps to users."""

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if "problem" in messages[0].content.lower():
            return Response(chat_message=TextMessage(content="Please follow these troubleshooting steps.", source=self.name))
        return await super().on_messages(messages, cancellation_token)


class TeamChat:
    """Integrates a team of agents, runs group chat scenarios in a round-robin order."""

    async def run_round_robin_chat(self, messages: List[BaseChatMessage], termination_condition: TerminationCondition):
        # Example agents working in a group chat
        agent1 = CustomerSupportAgent(
            "SupportAgent1", "Customer Support Agent 1")
        agent2 = HelpDeskAgent("HelpDeskAgent1", "Help Desk Agent 1")

        # Run team in a round-robin fashion
        team = RoundRobinGroupChat(
            [agent1, agent2], termination_condition=termination_condition)

        result = await team.run(task=messages)
        for message in result.messages:
            logger.info(f"Agent {message.source} responded: {message.content}")
        return result

    async def run(self):
        # Create messages to simulate real customer interaction
        messages = [
            TextMessage(content="I have an issue with my account",
                        source="user"),
            TextMessage(
                content="Can you help with a problem in the system?", source="user"),
        ]

        termination = MaxMessageTermination(4)  # Stop after 4 messages
        result = await self.run_round_robin_chat(messages, termination)

        logger.info(f"Chat completed with {len(result.messages)} messages.")


async def main():
    # Example use case of running group chat
    chat_system = TeamChat()
    await chat_system.run()

# To run the main function
if __name__ == "__main__":
    asyncio.run(main())
