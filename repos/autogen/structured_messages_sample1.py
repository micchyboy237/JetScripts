import json
from typing import List
from pydantic import BaseModel

# Import necessary message classes
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


class TestContent(BaseModel):
    """Test content model."""
    field1: str
    field2: int


def test_structured_message() -> dict:
    # Create a structured message with the test content
    message = StructuredMessage[TestContent](
        source="test_agent",
        content=TestContent(field1="test", field2=42),
    )

    # Prepare assertions and results
    result = {
        "message_type": message.type,
        "content_type": isinstance(message.content, TestContent),
        "content_fields": {
            "field1": message.content.field1,
            "field2": message.content.field2
        },
        "dumped_message": message.model_dump(),
    }

    return result


def test_message_factory() -> dict:
    factory = MessageFactory()

    # Text message data
    text_data = {
        "type": "TextMessage",
        "source": "test_agent",
        "content": "Hello, world!",
    }

    # Create a TextMessage instance
    text_message = factory.create(text_data)

    # Handoff message data
    handoff_data = {
        "type": "HandoffMessage",
        "source": "test_agent",
        "content": "handoff to another agent",
        "target": "target_agent",
    }

    # Create a HandoffMessage instance
    handoff_message = factory.create(handoff_data)

    # Structured message data
    structured_data = {
        "type": "StructuredMessage[TestContent]",
        "source": "test_agent",
        "content": {
            "field1": "test",
            "field2": 42,
        },
    }

    # Attempt to create StructuredMessage before registering
    try:
        factory.create(structured_data)
    except ValueError:
        factory.register(StructuredMessage[TestContent])
        structured_message = factory.create(structured_data)

    # Return results for verification
    result = {
        "text_message": {
            "type": text_message.type,
            "source": text_message.source,
            "content": text_message.content,
        },
        "handoff_message": {
            "type": handoff_message.type,
            "source": handoff_message.source,
            "content": handoff_message.content,
            "target": handoff_message.target,
        },
        "structured_message": {
            "type": structured_message.type,
            "content_field1": structured_message.content.field1,
            "content_field2": structured_message.content.field2,
        },
    }

    return result


class TestContainer(BaseModel):
    chat_messages: List[ChatMessage]
    agent_events: List[AgentEvent]


def test_union_types() -> dict:
    # Create a few messages.
    chat_messages: List[ChatMessage] = [
        TextMessage(source="user", content="Hello!"),
        MultiModalMessage(source="user", content=["Hello!", "World!"]),
        HandoffMessage(
            source="user", content="handoff to another agent", target="target_agent"),
        StopMessage(source="user", content="stop"),
    ]

    # Create a few agent events.
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

    # Create a container with the messages.
    container = TestContainer(
        chat_messages=chat_messages, agent_events=agent_events)

    # Dump the container to JSON.
    data = container.model_dump()

    # Load the container from JSON.
    loaded_container = TestContainer.model_validate(data)

    # Return results for verification
    result = {
        "chat_messages": chat_messages,
        "agent_events": agent_events,
        "loaded_chat_messages": loaded_container.chat_messages,
        "loaded_agent_events": loaded_container.agent_events,
    }

    return result


def main():
    # Run tests and print results
    structured_message_result = test_structured_message()
    print("Structured Message Test Result:", structured_message_result)

    message_factory_result = test_message_factory()
    print("Message Factory Test Result:", message_factory_result)

    union_types_result = test_union_types()
    print("Union Types Test Result:", union_types_result)


if __name__ == "__main__":
    main()
