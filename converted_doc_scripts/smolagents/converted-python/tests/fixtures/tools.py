from jet.logger import logger
from smolagents.tools import Tool, tool
import os
import pytest
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




@pytest.fixture
def test_tool():
    class TestTool(Tool):
        name = "test_tool"
        description = "A test tool"
        inputs = {"input": {"type": "string", "description": "Input value"}}
        output_type = "string"

        def forward(self, input):
            if input == "error":
                raise ValueError("Tool execution error")
            return f"Processed: {input}"

    return TestTool()


@pytest.fixture
def no_input_tool():
    class NoInputTool(Tool):
        name = "no_input_tool"
        description = "Tool with no inputs"
        inputs = {}
        output_type = "string"

        def forward(self):
            return "test"

    return NoInputTool()


@pytest.fixture
def single_input_tool():
    class SingleInputTool(Tool):
        name = "single_input_tool"
        description = "Tool with one input"
        inputs = {"text": {"type": "string", "description": "Input text"}}
        output_type = "string"

        def forward(self, text):
            return "test"

    return SingleInputTool()


@pytest.fixture
def multi_input_tool():
    class MultiInputTool(Tool):
        name = "multi_input_tool"
        description = "Tool with multiple inputs"
        inputs = {
            "text": {"type": "string", "description": "Text input"},
            "count": {"type": "integer", "description": "Number count"},
        }
        output_type = "object"

        def forward(self, text, count):
            return "test"

    return MultiInputTool()


@pytest.fixture
def multiline_description_tool():
    class MultilineDescriptionTool(Tool):
        name = "multiline_description_tool"
        description = "This is a tool with\nmultiple lines\nin the description"
        inputs = {"input": {"type": "string", "description": "Some input"}}
        output_type = "string"

        def forward(self, input):
            return "test"

    return MultilineDescriptionTool()


@pytest.fixture
def example_tool():
    @tool
    def valid_tool_function(input: str) -> str:
        """A valid tool function.

        Args:
            input (str): Input string.
        """
        return input.upper()

    return valid_tool_function


@pytest.fixture
def boolean_default_tool_class():
    class BooleanDefaultTool(Tool):
        name = "boolean_default_tool"
        description = "A tool with a boolean default parameter"
        inputs = {
            "text": {"type": "string", "description": "Input text"},
            "flag": {"type": "boolean", "description": "Boolean flag with default value", "nullable": True},
        }
        output_type = "string"

        def forward(self, text: str, flag: bool = False) -> str:
            return f"Text: {text}, Flag: {flag}"

    return BooleanDefaultTool()


@pytest.fixture
def boolean_default_tool_function():
    @tool
    def boolean_default_tool(text: str, flag: bool = False) -> str:
        """
        A tool with a boolean default parameter.

        Args:
            text: Input text
            flag: Boolean flag with default value
        """
        return f"Text: {text}, Flag: {flag}"

    return boolean_default_tool


@pytest.fixture
def optional_input_tool_class():
    class OptionalInputTool(Tool):
        name = "optional_input_tool"
        description = "A tool with an optional input parameter"
        inputs = {
            "required_text": {"type": "string", "description": "Required input text"},
            "optional_text": {"type": "string", "description": "Optional input text", "nullable": True},
        }
        output_type = "string"

        def forward(self, required_text: str, optional_text: str | None = None) -> str:
            if optional_text:
                return f"{required_text} + {optional_text}"
            return required_text

    return OptionalInputTool()


@pytest.fixture
def optional_input_tool_function():
    @tool
    def optional_input_tool(required_text: str, optional_text: str | None = None) -> str:
        """
        A tool with an optional input parameter.

        Args:
            required_text: Required input text
            optional_text: Optional input text
        """
        if optional_text:
            return f"{required_text} + {optional_text}"
        return required_text

    return optional_input_tool

logger.info("\n\n[DONE]", bright=True)