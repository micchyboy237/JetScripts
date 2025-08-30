from haystack import Pipeline
from haystack import component
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.tools import ToolInvoker
from haystack.core.component.types import Variadic
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from jet.logger import CustomLogger
from rich import print
from typing import Any, Dict, List
from typing import List
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🛠️ Define & Run Tools

In this notebook, we introduce the features we've developed for tool/function calling support in Haystack.

- We refactored the `ChatMessage` dataclass, to be more flexible and future-proof.
- We introduced some new dataclasses: `ToolCall`, `ToolCallResult`, and `Tool`.
- We added support for tools in the `OllamaFunctionCallingAdapterChatGenerator` and other Chat Generators.
- We introduced the `ToolInvoker` component, to actually execute tool calls prepared by Language Models.

We will first introduce the new features and then show two examples:
- A basic assistant that can answer user questions by either using a weather tool or relying on its own knowledge.
- A refined version of the assistant that can process the tool's output further before presenting it to the user.

For a more complex example, involving multiple tools and a Human-in-the-Loop interaction, check out this tutorial: [Building a Chat Agent with Function Calling](https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling).
"""
logger.info("# 🛠️ Define & Run Tools")

# ! pip install haystack-ai "sentence-transformers>=3.0.0"

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#   os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCallingAdapter API key:")

"""
## New experimental features

### Refactored `ChatMessage` dataclass, `ToolCall`, and `ToolCallResult`

The `ChatMessage` dataclass has been updated so that the `content` field is no longer just a string (`str`); it is now a list that can hold various types, including `TextContent`, `ToolCall`, and `ToolCallResult`.

The class methods `from_user`, `from_system`, `from_assistant`, and `from_tool` (newly added) are the recommended ways to create `ChatMessage` instances.

Additionally, we introduced:
- `ToolCall` dataclass: represents a tool call prepared by a Language Model.
- `ToolCallResult` dataclass: represents the result of a tool invocation.

Let's see some examples.
"""
logger.info("## New experimental features")


user_message = ChatMessage.from_user("What is the capital of Australia?")
logger.debug(user_message)

logger.debug(user_message.text)
logger.debug(user_message.texts)

logger.debug(user_message.tool_call)
logger.debug(user_message.tool_calls)

logger.debug(user_message.tool_call_result)
logger.debug(user_message.tool_call_results)

assistant_message = ChatMessage.from_assistant("How can I assist you today?")
logger.debug(assistant_message)

logger.debug(assistant_message.text)
logger.debug(assistant_message.texts)

logger.debug(assistant_message.tool_call)
logger.debug(assistant_message.tool_calls)

logger.debug(assistant_message.tool_call_result)
logger.debug(assistant_message.tool_call_results)

tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Rome"})

assistant_message_w_tool_call = ChatMessage.from_assistant(tool_calls=[tool_call])

logger.debug(assistant_message_w_tool_call.text)
logger.debug(assistant_message_w_tool_call.texts)

logger.debug(assistant_message_w_tool_call.tool_call)
logger.debug(assistant_message_w_tool_call.tool_calls)

logger.debug(assistant_message_w_tool_call.tool_call_result)
logger.debug(assistant_message_w_tool_call.tool_call_results)

tool_message = ChatMessage.from_tool(tool_result="temperature: 25°C", origin=tool_call, error=False)

logger.debug(tool_message.text)
logger.debug(tool_message.texts)

logger.debug(tool_message.tool_call)
logger.debug(tool_message.tool_calls)

logger.debug(tool_message.tool_call_result)
logger.debug(tool_message.tool_call_results)

"""
### `Tool` dataclass

This represents a tool for which Language Models can prepare a call.

It has the following attributes:
- `name`
- `description`
- `parameters`: a JSON schema describing the expected parameters
- `function`: a callable that is invoked when the tool is called

Accurate definitions of the textual attributes such as `name` and `description` are important for the Language Model to correctly prepare the call.

`Tool` exposes a `tool_spec` property, returning the tool specification to be used by Language Models.

It also has an `invoke` method that executes the underlying function with the provided parameters.

Let's see an example.
"""
logger.info("### `Tool` dataclass")


def add(a: int, b: int) -> int:
    return a + b


parameters = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"}
    },
    "required": ["a", "b"]
}

add_tool = Tool(name="addition_tool",
            description="This tool adds two numbers",
            parameters=parameters,
            function=add)

logger.debug(add_tool.tool_spec)

logger.debug(add_tool.invoke(a=15, b=10))

"""
### Support for tools in `OllamaFunctionCallingAdapterChatGenerator`

The `OllamaFunctionCallingAdapterChatGenerator` now supports tools. You can pass tools during initialization or via the `run` method, and it will use them to prepare tool calls when appropriate.

Here are some examples.
"""
logger.info("### Support for tools in `OllamaFunctionCallingAdapterChatGenerator`")


chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[add_tool])

res=chat_generator.run([ChatMessage.from_user("10 + 238")])
logger.debug(res)

res=chat_generator.run([ChatMessage.from_user("What is the habitat of a lion?")])
logger.debug(res)

chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

res=chat_generator.run([ChatMessage.from_user("10 + 238")])
logger.debug(res)

res_w_tool_call=chat_generator.run([ChatMessage.from_user("10 + 238")], tools=[add_tool])
logger.debug(res_w_tool_call)

"""
### `ToolInvoker` component

This component is responsible for executing tool calls prepared by Language Models.
It expects a list of messages (which may include tool calls) and returns a list of tool messages, containing the results of the tool invocations.
"""
logger.info("### `ToolInvoker` component")


tool_invoker = ToolInvoker(tools=[add_tool])

logger.debug(tool_invoker.run(res_w_tool_call["replies"]))

"""
`ToolInvoker` has 2 additional init parameters:
- `raise_on_failure`: if True, the component raises an exception in case of errors (tool not found, tool invocation errors, tool result conversion errors). Otherwise, it returns a `ChatMessage` object with `error=True` and a description of the error in `result`.
- `convert_result_to_json_string`: if True, the tool invocation result will be converted to a string using `json.dumps`. If False, converts the result using `str` (default).

Let's see how `raise_on_failure` works.
"""
logger.info("Let's see how `raise_on_failure` works.")

tool_call = ToolCall(tool_name="division_tool", arguments={"c": 1, "d": 2})

tool_invoker = ToolInvoker(tools=[add_tool], raise_on_failure=True)

tool_invoker.run([ChatMessage.from_assistant(tool_calls=[tool_call])])

tool_invoker = ToolInvoker(tools=[add_tool], raise_on_failure=False)

logger.debug(tool_invoker.run([ChatMessage.from_assistant(tool_calls=[tool_call])]))

"""
## End-to-end examples

In this section, we’ll put together everything we've covered so far into some practical, end-to-end examples.

### A simple use case

We'll start by creating a basic assistant that can answer user questions by either using a weather tool or relying on its own knowledge.
"""
logger.info("## End-to-end examples")


def dummy_weather(location: str):
    return {"temp": f"{random.randint(-10, 40)} °C",
            "humidity": f"{random.randint(0, 100)}%"}


weather_tool = Tool(
    name="weather",
    description="A tool to get the weather",
    function=dummy_weather,
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)


chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[weather_tool])

tool_invoker = ToolInvoker(tools=[weather_tool])

user_message = ChatMessage.from_user("What is the weather in Berlin?")

replies = chat_generator.run(messages=[user_message])["replies"]
logger.debug(f"assistant messages: {replies}")

if replies[0].tool_calls:
    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
    logger.debug(f"tool messages: {tool_messages}")

"""
The assistant correctly identifies when a tool is needed to answer a question and calls the appropriate tool.
"""
logger.info("The assistant correctly identifies when a tool is needed to answer a question and calls the appropriate tool.")

user_message = ChatMessage.from_user("What is the capital of Australia?")

replies = chat_generator.run(messages=[user_message])["replies"]
logger.debug(f"assistant messages: {replies}")

if replies[0].tool_calls:
    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
    logger.debug(f"tool messages: {tool_messages}")

"""
This time, the assistant uses its internal knowledge to answer the question.

#### Using a Pipeline

To achieve similar functionality using a Pipeline, we'll introduce a [Conditional Router](https://docs.haystack.deepset.ai/docs/conditionalrouter) that directs the flow based on whether the reply contain a tool call or not.
"""
logger.info("#### Using a Pipeline")


routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "{{replies}}",
        "output_name": "there_are_tool_calls",
        "output_type": List[ChatMessage],
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "{{replies}}",
        "output_name": "final_replies",
        "output_type": List[ChatMessage],
    },
]

tools_pipe = Pipeline()
tools_pipe.add_component("generator", OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[weather_tool]))
tools_pipe.add_component("router", ConditionalRouter(routes, unsafe=True))
tools_pipe.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))


tools_pipe.connect("generator.replies", "router")
tools_pipe.connect("router.there_are_tool_calls", "tool_invoker")

tools_pipe.show()

res=tools_pipe.run({"messages":[ChatMessage.from_user("What is the capital of Australia?")]})
logger.debug(res)

logger.debug("-"*50)

res=tools_pipe.run({"messages":[ChatMessage.from_user("What is the weather in Berlin?")]})
logger.debug(res)

"""
In this example, in case of tool calls, we end up with the raw tool invocation result, wrapped in a `ChatMessage` from tool role.
In the next example, we will see how to process this result further.

### Processing tool results with the Chat Generator

Depending on our use case and the tools involved, we might want to further process the tool's output before presenting it to the user. This can make the response more user-friendly.

In the next example, we'll pass the tool's response back to the Chat Generator for final processing.
"""
logger.info("### Processing tool results with the Chat Generator")

chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[weather_tool])
tool_invoker = ToolInvoker(tools=[weather_tool])

user_message = ChatMessage.from_user("What is the weather in Berlin?")

replies = chat_generator.run(messages=[user_message])["replies"]
logger.debug(f"assistant messages: {replies}")

if replies[0].tool_calls:

    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
    logger.debug(f"tool messages: {tool_messages}")

    messages = [user_message] + replies + tool_messages
    final_replies = chat_generator.run(messages=messages)["replies"]
    logger.debug(f"final assistant messages: {final_replies}")

"""
The assistant refines the tool's output to create a more human-readable response.

#### Using a Pipeline

The Pipeline is similar to the previous one.

We introduce a custom component, `MessageCollector`, to temporarily store the messages.
"""
logger.info("#### Using a Pipeline")


@component()
class MessageCollector:
    def __init__(self):
        self._messages = []

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: Variadic[List[ChatMessage]]) -> Dict[str, Any]:

        self._messages.extend([msg for inner in messages for msg in inner])
        return {"messages": self._messages}

    def clear(self):
        self._messages = []

message_collector = MessageCollector()

routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "{{replies}}",
        "output_name": "there_are_tool_calls",
        "output_type": List[ChatMessage],
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "{{replies}}",
        "output_name": "final_replies",
        "output_type": List[ChatMessage],
    },
]

tool_agent = Pipeline()
tool_agent.add_component("message_collector", message_collector)
tool_agent.add_component("generator", OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[weather_tool]))
tool_agent.add_component("router", ConditionalRouter(routes, unsafe=True))
tool_agent.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))


tool_agent.connect("message_collector", "generator.messages")
tool_agent.connect("generator.replies", "router")
tool_agent.connect("router.there_are_tool_calls", "tool_invoker")
tool_agent.connect("router.there_are_tool_calls", "message_collector")
tool_agent.connect("tool_invoker.tool_messages", "message_collector")

tool_agent.show()

message_collector.clear()
res=tool_agent.run({"messages":[ChatMessage.from_user("What is the capital of Australia?")]})
logger.debug(res)

logger.debug("-"*50)

message_collector.clear()
res=tool_agent.run({"messages":[ChatMessage.from_user("What is the weather in Berlin?")]})
logger.debug(res)

logger.debug("-"*50)

message_collector.clear()
res=tool_agent.run({"messages":[ChatMessage.from_user("What is the weather in Rome and Bangkok?")]})
logger.debug(res)

"""
For a more complex example, involving multiple tools and a Human-in-the-Loop interaction, check out this tutorial: [Building a Chat Agent with Function Calling](https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling).
"""
logger.info("For a more complex example, involving multiple tools and a Human-in-the-Loop interaction, check out this tutorial: [Building a Chat Agent with Function Calling](https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling).")

logger.info("\n\n[DONE]", bright=True)