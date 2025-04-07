import asyncio
import logging
from typing import Sequence

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, StopMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import AgentRuntime, SingleThreadedAgentRuntime
from autogen_ext.models.replay import ReplayChatCompletionClient

# Set up logger
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)


class _RealWorldAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._last_message: str | None = None
        self._total_messages = 0

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatAgent]]:
        return (TextMessage,)

    @property
    def total_messages(self) -> int:
        return self._total_messages

    async def on_messages(self, messages: Sequence[BaseChatAgent], cancellation_token: str) -> Response:
        if len(messages) > 0:
            self._last_message = messages[0].content
            self._total_messages += 1
            return Response(chat_message=TextMessage(content=messages[0].content, source=self.name))
        else:
            return Response(chat_message=TextMessage(content="No new messages.", source=self.name))

    async def on_reset(self, cancellation_token: str) -> None:
        self._last_message = None


# Create a group chat with agents based on conditions
async def real_world_group_chat_example(runtime: AgentRuntime) -> None:
    model_client = ReplayChatCompletionClient(["assistant_agent"])

    # Define a group of agents with descriptions
    agent1 = _RealWorldAgent(
        "Agent 1", description="Handles product inquiries")
    agent2 = _RealWorldAgent("Agent 2", description="Handles customer service")
    agent3 = _RealWorldAgent("Agent 3", description="Handles billing issues")

    # Define a function that selects the agent based on the last message
    def _select_agent(messages: Sequence[BaseChatAgent]) -> str | None:
        if len(messages) == 0:
            return "Agent 1"
        elif messages[-1].source == "Agent 1":
            return "Agent 2"
        elif messages[-1].source == "Agent 2":
            return "Agent 3"
        else:
            return "Agent 1"

    termination = MaxMessageTermination(5)

    # Create the group chat instance
    team = SelectorGroupChat(
        participants=[agent1, agent2, agent3],
        model_client=model_client,
        selector_func=_select_agent,
        termination_condition=termination,
        runtime=runtime,
    )

    # Run the team interaction
    result = await team.run(task="customer support flow")
    logger.info(f"Resulting messages: {len(result.messages)}")
    for message in result.messages:
        logger.info(f"Message from {message.source}: {message.content}")
    logger.info(f"Stop reason: {result.stop_reason}")


# Example of agent failure and error handling
async def flaky_agent_example() -> None:
    try:
        # Simulate a flaky agent behavior
        agent = _RealWorldAgent(
            "Flaky Agent", description="A simulated flaky agent")
        await agent.on_messages([TextMessage(content="Test message", source="user")], cancellation_token="1234")
    except Exception as e:
        logger.error(f"Error with agent: {e}")


# Main function to run multiple examples
async def main():
    # Set up a runtime for the agents
    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        # Running the real world group chat scenario
        await real_world_group_chat_example(runtime)

        # Running an example with a flaky agent
        await flaky_agent_example()

    finally:
        # Make sure to stop the runtime properly
        await runtime.stop()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
