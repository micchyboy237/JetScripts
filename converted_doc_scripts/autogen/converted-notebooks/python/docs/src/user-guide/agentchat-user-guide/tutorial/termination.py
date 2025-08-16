import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage, ToolCallExecutionEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import Component
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
from pydantic import BaseModel
from typing import Sequence
from typing_extensions import Self
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Termination 

In the previous section, we explored how to define agents, and organize them into teams that can solve tasks. However, a run can go on forever, and in many cases, we need to know _when_ to stop them. This is the role of the termination condition.

AgentChat supports several termination condition by providing a base {py:class}`~autogen_agentchat.base.TerminationCondition` class and several implementations that inherit from it.

A termination condition is a callable that takes a sequence of {py:class}`~autogen_agentchat.messages.BaseAgentEvent` or {py:class}`~autogen_agentchat.messages.BaseChatMessage` objects **since the last time the condition was called**, and returns a {py:class}`~autogen_agentchat.messages.StopMessage` if the conversation should be terminated, or `None` otherwise.
Once a termination condition has been reached, it must be reset by calling {py:meth}`~autogen_agentchat.base.TerminationCondition.reset` before it can be used again.

Some important things to note about termination conditions: 
- They are stateful but reset automatically after each run ({py:meth}`~autogen_agentchat.base.TaskRunner.run` or {py:meth}`~autogen_agentchat.base.TaskRunner.run_stream`) is finished.
- They can be combined using the AND and OR operators.

```{note}
For group chat teams (i.e., {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`,
{py:class}`~autogen_agentchat.teams.SelectorGroupChat`, and {py:class}`~autogen_agentchat.teams.Swarm`),
the termination condition is called after each agent responds.
While a response may contain multiple inner messages, the team calls its termination condition just once for all the messages from a single response.
So the condition is called with the "delta sequence" of messages since the last time it was called.
```

Built-In Termination Conditions: 
1. {py:class}`~autogen_agentchat.conditions.MaxMessageTermination`: Stops after a specified number of messages have been produced, including both agent and task messages.
2. {py:class}`~autogen_agentchat.conditions.TextMentionTermination`: Stops when specific text or string is mentioned in a message (e.g., "TERMINATE").
3. {py:class}`~autogen_agentchat.conditions.TokenUsageTermination`: Stops when a certain number of prompt or completion tokens are used. This requires the agents to report token usage in their messages.
4. {py:class}`~autogen_agentchat.conditions.TimeoutTermination`: Stops after a specified duration in seconds.
5. {py:class}`~autogen_agentchat.conditions.HandoffTermination`: Stops when a handoff to a specific target is requested. Handoff messages can be used to build patterns such as {py:class}`~autogen_agentchat.teams.Swarm`. This is useful when you want to pause the run and allow application or user to provide input when an agent hands off to them.
6. {py:class}`~autogen_agentchat.conditions.SourceMatchTermination`: Stops after a specific agent responds.
7. {py:class}`~autogen_agentchat.conditions.ExternalTermination`: Enables programmatic control of termination from outside the run. This is useful for UI integration (e.g., "Stop" buttons in chat interfaces).
8. {py:class}`~autogen_agentchat.conditions.StopMessageTermination`: Stops when a {py:class}`~autogen_agentchat.messages.StopMessage` is produced by an agent.
9. {py:class}`~autogen_agentchat.conditions.TextMessageTermination`: Stops when a {py:class}`~autogen_agentchat.messages.TextMessage` is produced by an agent.
10. {py:class}`~autogen_agentchat.conditions.FunctionCallTermination`: Stops when a {py:class}`~autogen_agentchat.messages.ToolCallExecutionEvent` containing a {py:class}`~autogen_core.models.FunctionExecutionResult` with a matching name is produced by an agent.
11. {py:class}`~autogen_agentchat.conditions.FunctionalTermination`: Stop when a function expression is evaluated to `True` on the last delta sequence of messages. This is useful for quickly create custom termination conditions that are not covered by the built-in ones.

## Basic Usage

To demonstrate the characteristics of termination conditions, we'll create a team consisting of two agents: a primary agent responsible for text generation and a critic agent that reviews and provides feedback on the generated text.
"""
logger.info("# Termination")


model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
    temperature=1,
)

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback for every message. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

"""
Let's explore how termination conditions automatically reset after each `run` or `run_stream` call, allowing the team to resume its conversation from where it left off.
"""
logger.info("Let's explore how termination conditions automatically reset after each `run` or `run_stream` call, allowing the team to resume its conversation from where it left off.")

max_msg_termination = MaxMessageTermination(max_messages=3)
round_robin_team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=max_msg_termination)

async def run_async_code_f4dc020b():
    await Console(round_robin_team.run_stream(task="Write a unique, Haiku about the weather in Paris"))
    return 
 = asyncio.run(run_async_code_f4dc020b())
logger.success(format_json())

"""
The conversation stopped after reaching the maximum message limit. Since the primary agent didn't get to respond to the feedback, let's continue the conversation.
"""
logger.info("The conversation stopped after reaching the maximum message limit. Since the primary agent didn't get to respond to the feedback, let's continue the conversation.")

async def run_async_code_2f58f4ee():
    await Console(round_robin_team.run_stream())
    return 
 = asyncio.run(run_async_code_2f58f4ee())
logger.success(format_json())

"""
The team continued from where it left off, allowing the primary agent to respond to the feedback.

## Combining Termination Conditions

Let's show how termination conditions can be combined using the AND (`&`) and OR (`|`) operators to create more complex termination logic. For example, we'll create a team that stops either after 10 messages are generated or when the critic agent approves a message.
"""
logger.info("## Combining Termination Conditions")

max_msg_termination = MaxMessageTermination(max_messages=10)
text_termination = TextMentionTermination("APPROVE")
combined_termination = max_msg_termination | text_termination

round_robin_team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=combined_termination)

async def run_async_code_f4dc020b():
    await Console(round_robin_team.run_stream(task="Write a unique, Haiku about the weather in Paris"))
    return 
 = asyncio.run(run_async_code_f4dc020b())
logger.success(format_json())

"""
The conversation stopped after the critic agent approved the message, although it could have also stopped if 10 messages were generated.

Alternatively, if we want to stop the run only when both conditions are met, we can use the AND (`&`) operator.
"""
logger.info("The conversation stopped after the critic agent approved the message, although it could have also stopped if 10 messages were generated.")

combined_termination = max_msg_termination & text_termination

"""
## Custom Termination Condition

The built-in termination conditions are sufficient for most use cases.
However, there may be cases where you need to implement a custom termination condition that doesn't fit into the existing ones.
You can do this by subclassing the {py:class}`~autogen_agentchat.base.TerminationCondition` class.

In this example, we create a custom termination condition that stops the conversation when
a specific function call is made.
"""
logger.info("## Custom Termination Condition")




class FunctionCallTerminationConfig(BaseModel):
    """Configuration for the termination condition to allow for serialization
    and deserialization of the component.
    """

    function_name: str


class FunctionCallTermination(TerminationCondition, Component[FunctionCallTerminationConfig]):
    """Terminate the conversation if a FunctionExecutionResult with a specific name is received."""

    component_config_schema = FunctionCallTerminationConfig
    component_provider_override = "autogen_agentchat.conditions.FunctionCallTermination"
    """The schema for the component configuration."""

    def __init__(self, function_name: str) -> None:
        self._terminated = False
        self._function_name = function_name

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for message in messages:
            if isinstance(message, ToolCallExecutionEvent):
                for execution in message.content:
                    if execution.name == self._function_name:
                        self._terminated = True
                        return StopMessage(
                            content=f"Function '{self._function_name}' was executed.",
                            source="FunctionCallTermination",
                        )
        return None

    async def reset(self) -> None:
        self._terminated = False

    def _to_config(self) -> FunctionCallTerminationConfig:
        return FunctionCallTerminationConfig(
            function_name=self._function_name,
        )

    @classmethod
    def _from_config(cls, config: FunctionCallTerminationConfig) -> Self:
        return cls(
            function_name=config.function_name,
        )

"""
Let's use this new termination condition to stop the conversation when the critic agent approves a message
using the `approve` function call.

First we create a simple function that will be called when the critic agent approves a message.
"""
logger.info("Let's use this new termination condition to stop the conversation when the critic agent approves a message")

def approve() -> None:
    """Approve the message when all feedbacks have been addressed."""
    pass

"""
Then we create the agents. The critic agent is equipped with the `approve` tool.
"""
logger.info("Then we create the agents. The critic agent is equipped with the `approve` tool.")


model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
    temperature=1,
)

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    tools=[approve],  # Register the approve function as a tool.
    system_message="Provide constructive feedback. Use the approve tool to approve when all feedbacks are addressed.",
)

"""
Now, we create the termination condition and the team.
We run the team with the poem-writing task.
"""
logger.info("Now, we create the termination condition and the team.")

function_call_termination = FunctionCallTermination(function_name="approve")
round_robin_team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=function_call_termination)

async def run_async_code_f4dc020b():
    await Console(round_robin_team.run_stream(task="Write a unique, Haiku about the weather in Paris"))
    return 
 = asyncio.run(run_async_code_f4dc020b())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
You can see that the conversation stopped when the critic agent approved the message using the `approve` function call.
"""
logger.info("You can see that the conversation stopped when the critic agent approved the message using the `approve` function call.")

logger.info("\n\n[DONE]", bright=True)