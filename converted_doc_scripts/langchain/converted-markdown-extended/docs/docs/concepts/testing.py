from jet.logger import logger
from langchain_core.messages import AIMessage, ToolCall, convert_to_ollama_messages
from langchain_tests.unit_tests import ToolsUnitTests
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Testing
<span data-heading-keywords="tests,testing,unit,integration"></span>

Testing is a critical part of the development process that ensures your code works as expected and meets the desired quality standards.

In the LangChain ecosystem, we have 2 main types of tests: **unit tests** and **integration tests**.

For integrations that implement standard LangChain abstractions, we have a set of **standard tests** (both unit and integration) that help maintain compatibility between different components and ensure reliability of high-usage ones.

## Unit Tests

**Definition**: Unit tests are designed to validate the smallest parts of your code—individual functions or methods—ensuring they work as expected in isolation. They do not rely on external systems or integrations.

**Example**: Testing the `convert_langchain_aimessage_to_dict` function to confirm it correctly converts an AI message to a dictionary format:
"""
logger.info("# Testing")


def test_convert_to_ollama_messages():
    ai_message = AIMessage(
        content="Let me call that tool for you!",
        tool_calls=[
            ToolCall(name='parrot_multiply_tool', id='1', args={'a': 2, 'b': 3}),
        ]
    )

    result = convert_to_ollama_messages(ai_message)

    expected = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "1",
                "function": {
                    "name": "parrot_multiply_tool",
                    "arguments": '{"a": 2, "b": 3}',
                },
            }
        ],
        "content": "Let me call that tool for you!",
    }
    assert result == expected  # Ensure conversion matches expected output

"""
---

## Integration Tests

**Definition**: Integration tests validate that multiple components or systems work together as expected. For tools or integrations relying on external services, these tests often ensure end-to-end functionality.

**Example**: Testing `ParrotMultiplyTool` with access to an API service that multiplies two numbers and adds 80:
"""
logger.info("## Integration Tests")

def test_integration_with_service():
    tool = ParrotMultiplyTool()
    result = tool.invoke({"a": 2, "b": 3})
    assert result == 86

"""
---

## Standard Tests

**Definition**: Standard tests are pre-defined tests provided by LangChain to ensure consistency and reliability across all tools and integrations. They include both unit and integration test templates tailored for LangChain components.

**Example**: Subclassing LangChain's `ToolsUnitTests` or `ToolsIntegrationTests` to automatically run standard tests:
"""
logger.info("## Standard Tests")


class TestParrotMultiplyToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ParrotMultiplyTool

    def tool_invoke_params_example(self):
        return {"a": 2, "b": 3}

"""
To learn more, check out our guide on [how to add standard tests to an integration](../../contributing/how_to/integrations/standard_tests).
"""
logger.info("To learn more, check out our guide on [how to add standard tests to an integration](../../contributing/how_to/integrations/standard_tests).")

logger.info("\n\n[DONE]", bright=True)