import json
from typing import List
from jet.logger import logger
from jet.transformers.formatters import format_json
from pydantic import BaseModel
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MessageFactory,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    StopMessage,
    StructuredMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult


# Real-world Example 1: Structured Message Creation
class TestContent(BaseModel):
    """Test content model."""

    field1: str
    field2: int


def create_structured_message() -> StructuredMessage[TestContent]:
    message = StructuredMessage[TestContent](
        source="test_agent",
        content=TestContent(field1="Welcome", field2=100),
    )

    # Log the result
    logger.log(
        f"StructuredMessage created:", format_json(message.model_dump()), colors=["DEBUG", "SUCCESS"])
    return message


# Real-world Example 2: Message Factory Usage
def create_text_message() -> TextMessage:
    factory = MessageFactory()
    text_data = {
        "type": "TextMessage",
        "source": "test_agent",
        "content": "Hello, user!",
    }

    # Create the TextMessage using the factory
    text_message = factory.create(text_data)
    logger.log(
        f"TextMessage created:", format_json(text_message.model_dump()), colors=["DEBUG", "SUCCESS"])
    return text_message


def create_handoff_message() -> HandoffMessage:
    factory = MessageFactory()
    handoff_data = {
        "type": "HandoffMessage",
        "source": "test_agent",
        "content": "handoff to support agent",
        "target": "support_agent",
    }

    # Create the HandoffMessage using the factory
    handoff_message = factory.create(handoff_data)
    logger.log(
        f"HandoffMessage created:", format_json(handoff_message.model_dump()), colors=["DEBUG", "SUCCESS"])
    return handoff_message


# Real-world Example 3: Union Types with Messages and Events
class TestContainer(BaseModel):
    chat_messages: List[ChatMessage]
    agent_events: List[AgentEvent]


def create_union_types() -> TestContainer:
    chat_messages: List[ChatMessage] = [
        TextMessage(source="user", content="Hello!"),
        MultiModalMessage(source="user", content=["Hello!", "World!"]),
        HandoffMessage(
            source="user", content="handoff to another agent", target="support_agent"),
        StopMessage(source="user", content="stop"),
    ]

    agent_events: List[AgentEvent] = [
        ModelClientStreamingChunkEvent(source="user", content="Hello!"),
        ToolCallRequestEvent(
            content=[
                FunctionCall(id="1", name="test_function", arguments=json.dumps(
                    {"arg1": "value1", "arg2": "value2"}))
            ],
            source="user",
        ),
        ToolCallExecutionEvent(
            content=[FunctionExecutionResult(call_id="1", content="result", name="test")], source="user"
        ),
    ]

    # Create a container with the chat messages and agent events
    container = TestContainer(
        chat_messages=chat_messages, agent_events=agent_events)
    logger.log(
        f"TestContainer created:", format_json(container.model_dump()), colors=["DEBUG", "SUCCESS"])
    return container


# Real-world Example 4: Function Calls with Tool Execution
def create_tool_call_event() -> ToolCallRequestEvent:
    tool_call_event = ToolCallRequestEvent(
        content=[
            FunctionCall(id="1", name="process_order", arguments=json.dumps(
                {"order_id": "123", "user_id": "456"}))
        ],
        source="user",
    )

    logger.log(
        f"ToolCallRequestEvent created:", format_json(tool_call_event.model_dump()), colors=["DEBUG", "SUCCESS"])
    return tool_call_event


def execute_tool_call() -> ToolCallExecutionEvent:
    tool_execution_event = ToolCallExecutionEvent(
        content=[FunctionExecutionResult(
            call_id="1", content="Order processed", name="process_order")],
        source="system",
    )

    logger.log(
        f"ToolCallExecutionEvent created:", format_json(tool_execution_event.model_dump()), colors=["DEBUG", "SUCCESS"])
    return tool_execution_event


# Main Function to Run All Examples
def main():
    # Example 1: Create Structured Message
    create_structured_message()

    # Example 2: Create Text and Handoff Messages using Message Factory
    create_text_message()
    create_handoff_message()

    # Example 3: Create TestContainer with Messages and Agent Events
    create_union_types()

    # Example 4: Create and Execute Tool Call Event
    create_tool_call_event()
    execute_tool_call()


if __name__ == "__main__":
    main()
