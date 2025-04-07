import asyncio
import logging
import json
from typing import Any, AsyncGenerator, List, Sequence

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import BaseChatMessage, TextMessage, HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_core import AgentRuntime, CancellationToken, FunctionCall, SingleThreadedAgentRuntime
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)


class OrderProcessingAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, next_agent: str) -> None:
        super().__init__(name, description)
        self._next_agent = next_agent

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if messages:
            order = messages[0].content
            logger.info(f"Processing order: {order}")
            return Response(chat_message=TextMessage(content=f"Order {order} is ready for shipment.", source=self.name))
        return Response(chat_message=TextMessage(content="No orders to process.", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class ShipmentAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, next_agent: str) -> None:
        super().__init__(name, description)
        self._next_agent = next_agent

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if messages:
            shipment = messages[0].content
            logger.info(f"Preparing shipment for: {shipment}")
            return Response(chat_message=TextMessage(content=f"Shipment for {shipment} is on the way!", source=self.name))
        return Response(chat_message=TextMessage(content="No shipment to prepare.", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


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
                content=f"Transferred to {self._next_agent}.", target=self._next_agent, source=self.name
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def main():
    # First, simulate an order being placed
    order_agent = OrderProcessingAgent(
        "order_agent", description="Processes incoming orders", next_agent="shipment_agent")
    shipment_agent = ShipmentAgent(
        "shipment_agent", description="Handles order shipment", next_agent="handoff_agent")

    # Set up agents in sequence for task delegation
    team = Swarm([order_agent, shipment_agent], max_turns=2,
                 runtime=SingleThreadedAgentRuntime())

    # Simulate a new order being placed
    order_task = "Order #12345 for electronics"
    result = await team.run(task=order_task)

    for message in result.messages:
        logger.info(
            f"Received message: {message.content} from {message.source}")

    # Add more task delegation
    handoff_agent = _HandOffAgent(
        "handoff_agent", description="Handles handoff to next step", next_agent="order_agent")
    team_with_handoff = Swarm([order_agent, shipment_agent, handoff_agent],
                              max_turns=3, runtime=SingleThreadedAgentRuntime())

    # Simulate order and shipment task delegation with handoff
    result = await team_with_handoff.run(task="Process order #67890")

    for message in result.messages:
        logger.info(
            f"Received message: {message.content} from {message.source}")

    # Log the completion message when done
    logger.info("All tasks processed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
