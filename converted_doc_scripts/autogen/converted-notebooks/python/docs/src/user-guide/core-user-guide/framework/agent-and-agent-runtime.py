import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OllamaChatCompletionClient
from dataclasses import dataclass
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent and Agent Runtime

In this and the following section, we focus on the core concepts of AutoGen:
agents, agent runtime, messages, and communication -- 
the foundational building blocks for an multi-agent applications.

```{note}
The Core API is designed to be unopinionated and flexible. So at times, you
may find it challenging. Continue if you are building
an interactive, scalable and distributed multi-agent system and want full control
of all workflows.
If you just want to get something running
quickly, you may take a look at the [AgentChat API](../../agentchat-user-guide/index.md).
```

An agent in AutoGen is an entity defined by the base interface {py:class}`~autogen_core.Agent`.
It has a unique identifier of the type {py:class}`~autogen_core.AgentId`,
a metadata dictionary of the type {py:class}`~autogen_core.AgentMetadata`.

In most cases, you can subclass your agents from higher level class {py:class}`~autogen_core.RoutedAgent` which enables you to route messages to corresponding message handler specified with {py:meth}`~autogen_core.message_handler` decorator and proper type hint for the `message` variable.
An agent runtime is the execution environment for agents in AutoGen.

Similar to the runtime environment of a programming language,
an agent runtime provides the necessary infrastructure to facilitate communication
between agents, manage agent lifecycles, enforce security boundaries, and support monitoring and
debugging.

For local development, developers can use {py:class}`~autogen_core.SingleThreadedAgentRuntime`,
which can be embedded in a Python application.

```{note}
Agents are not directly instantiated and managed by application code.
Instead, they are created by the runtime when needed and managed by the runtime.

If you are already familiar with [AgentChat](../../agentchat-user-guide/index.md),
it is important to note that AgentChat's agents such as
{py:class}`~autogen_agentchat.agents.AssistantAgent` are created by application 
and thus not directly managed by the runtime. To use an AgentChat agent in Core,
you need to create a wrapper Core agent that delegates messages to the AgentChat agent
and let the runtime manage the wrapper agent.
```

## Implementing an Agent

To implement an agent, the developer must subclass the {py:class}`~autogen_core.RoutedAgent` class
and implement a message handler method for each message type the agent is expected to handle using
the {py:meth}`~autogen_core.message_handler` decorator.
For example,
the following agent handles a simple message type `MyMessageType` and prints the message it receives:
"""
logger.info("# Agent and Agent Runtime")




@dataclass
class MyMessageType:
    content: str


class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        logger.debug(f"{self.id.type} received message: {message.content}")

"""
This agent only handles `MyMessageType` and messages will be delivered to `handle_my_message_type` method. Developers can have multiple message handlers for different message types by using {py:meth}`~autogen_core.message_handler` decorator and setting the type hint for the `message` variable in the handler function. You can also leverage [python typing union](https://docs.python.org/3/library/typing.html#typing.Union) for the `message` variable in one message handler function if it better suits agent's logic.
See the next section on [message and communication](./message-and-communication.ipynb).

## Using an AgentChat Agent

If you have an [AgentChat](../../agentchat-user-guide/index.md) agent and want to use it in the Core API, you can create
a wrapper {py:class}`~autogen_core.RoutedAgent` that delegates messages to the AgentChat agent.
The following example shows how to create a wrapper agent for the {py:class}`~autogen_agentchat.agents.AssistantAgent`
in AgentChat.
"""
logger.info("## Using an AgentChat Agent")



class MyAssistant(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        logger.debug(f"{self.id.type} received message: {message.content}")
        async def async_func_14():
            response = await self._delegate.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
            return response
        response = asyncio.run(async_func_14())
        logger.success(format_json(response))
        logger.debug(f"{self.id.type} responded: {response.chat_message}")

"""
For how to use model client, see the [Model Client](../components/model-clients.ipynb) section.

Since the Core API is unopinionated,
you are not required to use the AgentChat API to use the Core API.
You can implement your own agents or use another agent framework.

## Registering Agent Type

To make agents available to the runtime, developers can use the
{py:meth}`~autogen_core.BaseAgent.register` class method of the
{py:class}`~autogen_core.BaseAgent` class.
The process of registration associates an agent type, which is uniquely identified by a string, 
and a factory function
that creates an instance of the agent type of the given class.
The factory function is used to allow automatic creation of agent instances 
when they are needed.

Agent type ({py:class}`~autogen_core.AgentType`) is not the same as the agent class. In this example,
the agent type is `AgentType("my_agent")` or `AgentType("my_assistant")` and the agent class is the Python class `MyAgent` or `MyAssistantAgent`.
The factory function is expected to return an instance of the agent class 
on which the {py:meth}`~autogen_core.BaseAgent.register` class method is invoked.
Read [Agent Identity and Lifecycles](../core-concepts/agent-identity-and-lifecycle.md)
to learn more about agent type and identity.

```{note}
Different agent types can be registered with factory functions that return 
the same agent class. For example, in the factory functions, 
variations of the constructor parameters
can be used to create different instances of the same agent class.
```

To register our agent types with the 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
the following code can be used:
"""
logger.info("## Registering Agent Type")


runtime = SingleThreadedAgentRuntime()
async def run_async_code_6f198638():
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    return 
 = asyncio.run(run_async_code_6f198638())
logger.success(format_json())
async def run_async_code_c108382a():
    await MyAssistant.register(runtime, "my_assistant", lambda: MyAssistant("my_assistant"))
    return 
 = asyncio.run(run_async_code_c108382a())
logger.success(format_json())

"""
Once an agent type is registered, we can send a direct message to an agent instance
using an {py:class}`~autogen_core.AgentId`.
The runtime will create the instance the first time it delivers a
message to this instance.
"""
logger.info("Once an agent type is registered, we can send a direct message to an agent instance")

runtime.start()  # Start processing messages in the background.
async def run_async_code_a7edcc41():
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    return 
 = asyncio.run(run_async_code_a7edcc41())
logger.success(format_json())
async def run_async_code_46d144ab():
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_assistant", "default"))
    return 
 = asyncio.run(run_async_code_46d144ab())
logger.success(format_json())
async def run_async_code_32e9588a():
    await runtime.stop()  # Stop processing messages in the background.
    return 
 = asyncio.run(run_async_code_32e9588a())
logger.success(format_json())

"""
```{note}
Because the runtime manages the lifecycle of agents, an {py:class}`~autogen_core.AgentId`
is only used to communicate with the agent or retrieve its metadata (e.g., description).
```

## Running the Single-Threaded Agent Runtime

The above code snippet uses {py:meth}`~autogen_core.SingleThreadedAgentRuntime.start` to start a background task
to process and deliver messages to recepients' message handlers.
This is a feature of the
local embedded runtime {py:class}`~autogen_core.SingleThreadedAgentRuntime`.

To stop the background task immediately, use the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.stop` method:
"""
logger.info("## Running the Single-Threaded Agent Runtime")

runtime.start()
async def run_async_code_e8c0530d():
    await runtime.stop()  # This will return immediately but will not cancel
    return 
 = asyncio.run(run_async_code_e8c0530d())
logger.success(format_json())

"""
You can resume the background task by calling {py:meth}`~autogen_core.SingleThreadedAgentRuntime.start` again.

For batch scenarios such as running benchmarks for evaluating agents,
you may want to wait for the background task to stop automatically when
there are no unprocessed messages and no agent is handling messages --
the batch may considered complete.
You can achieve this by using the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.stop_when_idle` method:
"""
logger.info("You can resume the background task by calling {py:meth}`~autogen_core.SingleThreadedAgentRuntime.start` again.")

runtime.start()
async def run_async_code_28f4a243():
    await runtime.stop_when_idle()  # This will block until the runtime is idle.
    return 
 = asyncio.run(run_async_code_28f4a243())
logger.success(format_json())

"""
To close the runtime and release resources, use the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.close` method:
"""
logger.info("To close the runtime and release resources, use the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.close` method:")

async def run_async_code_aff70de8():
    await runtime.close()
    return 
 = asyncio.run(run_async_code_aff70de8())
logger.success(format_json())

"""
Other runtime implementations will have their own ways of running the runtime.
"""
logger.info("Other runtime implementations will have their own ways of running the runtime.")

logger.info("\n\n[DONE]", bright=True)