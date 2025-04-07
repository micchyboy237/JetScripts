import asyncio
import random
from dataclasses import dataclass
from typing import List

from autogen_agentchat.teams._group_chat._sequential_routed_agent import SequentialRoutedAgent
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)


@dataclass
class Message:
    content: str


@default_subscription
class TestAgent(SequentialRoutedAgent):
    def __init__(self, description: str) -> None:
        super().__init__(description=description,
                         sequential_message_types=[Message])
        self.messages: List[Message] = []

    @message_handler
    async def handle_content_publish(self, message: Message, ctx: MessageContext) -> None:
        await asyncio.sleep(random.random() / 100)
        self.messages.append(message)


async def run_sequential_routed_agent_sample(message_count: int = 100) -> List[Message]:
    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    await TestAgent.register(runtime, type="test_agent", factory=lambda: TestAgent(description="Test Agent"))
    test_agent_id = AgentId(type="test_agent", key="default")

    for i in range(message_count):
        await runtime.publish_message(Message(content=f"{i}"), topic_id=DefaultTopicId())

    await runtime.stop_when_idle()
    test_agent = await runtime.try_get_underlying_agent_instance(test_agent_id, TestAgent)

    return test_agent.messages


async def main():
    def assert_ordered_messages(messages, expected_count):
        for i in range(expected_count):
            expected = str(i)
            result = messages[i].content
            assert result == expected, f"Expected: {expected}, Got: {result}"
        print(f"✅ All {expected_count} messages are in order.")

    print("▶ Running sequential routed agent simulation...")
    message_count = 100
    messages = await run_sequential_routed_agent_sample(message_count=message_count)
    assert_ordered_messages(messages, message_count)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
