import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Type
from jet.llm.ollama.base import ChatResponse, Ollama
from jet.llm.tools.types import AsyncBaseTool, BaseTool, ToolMetadata, ToolOutput
from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from pydantic import BaseModel, Field
from enum import Enum
from abc import abstractmethod
from llama_index.core.tools.calling import call_tool

# Define the Pydantic schema for get_current_weather


class WeatherFormat(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetCurrentWeatherSchema(BaseModel):
    location: str = Field(
        ..., description="The location to get the weather for, e.g. San Francisco, CA"
    )
    format: WeatherFormat = Field(
        ..., description="The format to return the weather in, e.g. 'celsius' or 'fahrenheit'"
    )

# Define the get_current_weather function (synchronous)


def get_current_weather(location: str, format: WeatherFormat) -> str:
    """Get the current weather for a location."""
    if format == WeatherFormat.CELSIUS:
        temperature = 25
        symbol = "Celsius"
    else:
        temperature = 77
        symbol = "Fahrenheit"

    return f"The current weather in {location} is {temperature} degrees {symbol}."


# Define the OllamaTool class


class OllamaTool(BaseTool):
    def __init__(self, name: str, description: str, fn_schema: Type[BaseModel], fn: Callable):
        self._metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema,
            return_direct=False
        )
        self._fn = fn

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(self, input: Any) -> ToolOutput:
        try:
            # Validate input against the schema
            if isinstance(input, dict):
                validated_input = self.metadata.fn_schema(**input)
            else:
                validated_input = self.metadata.fn_schema(input=input)

            # Extract parameters
            params = validated_input.dict()
            result = self._fn(**params)

            return ToolOutput(
                content=str(result),
                tool_name=self.metadata.name,
                raw_input=params,
                raw_output=result,
                is_error=False
            )
        except Exception as e:
            return ToolOutput(
                content=str(e),
                tool_name=self.metadata.name,
                raw_input=input if isinstance(input, dict) else {
                    "input": input},
                raw_output=None,
                is_error=True
            )


def call_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
) -> ToolOutput:
    from llama_index.core.tools.calling import call_tool

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    output = call_tool(tool, {
        "input": tool_call.tool_kwargs
    })

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output

# Demonstration


def main():
    verbose = True
    # Create an OllamaTool instance for get_current_weather
    weather_tool = OllamaTool(
        name="get_current_weather",
        description="Get the current weather for a location",
        fn_schema=GetCurrentWeatherSchema,
        fn=get_current_weather
    )

    # Test synchronous call
    input_data = {"location": "San Francisco, CA", "format": "celsius"}
    result = weather_tool(input_data)
    logger.gray("Synchronous Result:")
    logger.success(result)
    # Output: content='The current weather in San Francisco, CA is 25° celsius.' ...

    # Generate Ollama tool JSON
    ollama_tool_json = weather_tool.metadata.to_ollama_tool()
    logger.gray("Ollama Tool JSON:")
    logger.success(format_json(ollama_tool_json))

    llm = Ollama(model="llama3.1")
    tools: Sequence[BaseTool] = [weather_tool]
    chat_response = llm.chat_with_tools(
        tools, user_msg="What is the weather today in Paris?")

    tool_calls: list[ToolSelection] = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=True
    )
    tool_outputs = [
        call_tool_with_selection(tool_call, tools, verbose=verbose)
        for tool_call in tool_calls
    ]
    tool_outputs_with_error = [
        tool_output for tool_output in tool_outputs if tool_output.is_error
    ]

    if len(tool_outputs_with_error) > 0:
        error_text = "\n\n".join(
            [tool_output.content for tool_output in tool_outputs]
        )
        raise ValueError(error_text)
    elif len(tool_outputs) == 1:
        chat_response.message = ChatMessage(
            role=chat_response.message.role,
            content=tool_outputs[0].content,
            additional_kwargs={},
        )

    logger.gray("Chat Response:")
    logger.success(chat_response.message.content)


if __name__ == "__main__":
    main()
