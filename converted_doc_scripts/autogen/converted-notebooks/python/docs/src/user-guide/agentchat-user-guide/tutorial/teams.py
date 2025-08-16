import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
from jet.logger import CustomLogger
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Teams

In this section you'll learn how to create a _multi-agent team_ (or simply team) using AutoGen. A team is a group of agents that work together to achieve a common goal.

We'll first show you how to create and run a team. We'll then explain how to observe the team's behavior, which is crucial for debugging and understanding the team's performance, and common operations to control the team's behavior.


AgentChat supports several team presets:

- {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`: A team that runs a group chat with participants taking turns in a round-robin fashion (covered on this page). [Tutorial](#creating-a-team) 
- {py:class}`~autogen_agentchat.teams.SelectorGroupChat`: A team that selects the next speaker using a ChatCompletion model after each message. [Tutorial](../selector-group-chat.ipynb)
- {py:class}`~autogen_agentchat.teams.MagenticOneGroupChat`: A  generalist multi-agent system for solving open-ended web and file-based tasks across a variety of domains. [Tutorial](../magentic-one.md) 
- {py:class}`~autogen_agentchat.teams.Swarm`: A team that uses {py:class}`~autogen_agentchat.messages.HandoffMessage` to signal transitions between agents. [Tutorial](../swarm.ipynb)

```{note}

**When should you use a team?**

Teams are for complex tasks that require collaboration and diverse expertise.
However, they also demand more scaffolding to steer compared to single agents.
While AutoGen simplifies the process of working with teams, start with
a single agent for simpler tasks, and transition to a multi-agent team when a single agent proves inadequate.
Ensure that you have optimized your single agent with the appropriate tools
and instructions before moving to a team-based approach.
```

## Creating a Team

{py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` is a simple yet effective team configuration where all agents share the same context and take turns responding in a round-robin fashion. Each agent, during its turn, broadcasts its response to all other agents, ensuring that the entire team maintains a consistent context.

We will begin by creating a team with two {py:class}`~autogen_agentchat.agents.AssistantAgent` and a {py:class}`~autogen_agentchat.conditions.TextMentionTermination` condition that stops the team when a specific word is detected in the agent's response.

The two-agent team implements the _reflection_ pattern, a multi-agent design pattern where a critic agent evaluates the responses of a primary agent. Learn more about the reflection pattern using the [Core API](../../core-user-guide/design-patterns/reflection.ipynb).
"""
logger.info("# Teams")



model_client = MLXChatCompletionClient(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

text_termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

"""
## Running a Team

Let's call the {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` method
to start the team with a task.
"""
logger.info("## Running a Team")

async def run_async_code_9817a56b():
    async def run_async_code_c3be6668():
        result = await team.run(task="Write a short poem about the fall season.")
        return result
    result = asyncio.run(run_async_code_c3be6668())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_9817a56b())
logger.success(format_json(result))
logger.debug(result)

"""
The team runs the agents until the termination condition was met.
In this case, the team ran agents following a round-robin order until the the
termination condition was met when the word "APPROVE" was detected in the
agent's response.
When the team stops, it returns a {py:class}`~autogen_agentchat.base.TaskResult` object with all the messages produced by the agents in the team.

## Observing a Team

Similar to the agent's {py:meth}`~autogen_agentchat.agents.BaseChatAgent.on_messages_stream` method, you can stream the team's messages while it is running by calling the {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream` method. This method returns a generator that yields messages produced by the agents in the team as they are generated, with the final item being the {py:class}`~autogen_agentchat.base.TaskResult` object.
"""
logger.info("## Observing a Team")

async def run_async_code_12a0d807():
    await team.reset()  # Reset the team for a new task.
    return 
 = asyncio.run(run_async_code_12a0d807())
logger.success(format_json())
async for message in team.run_stream(task="Write a short poem about the fall season."):  # type: ignore
    if isinstance(message, TaskResult):
        logger.debug("Stop Reason:", message.stop_reason)
    else:
        logger.debug(message)

"""
As demonstrated in the example above, you can determine the reason why the team stopped by checking the {py:attr}`~autogen_agentchat.base.TaskResult.stop_reason` attribute.

The {py:meth}`~autogen_agentchat.ui.Console` method provides a convenient way to print messages to the console with proper formatting.
"""
logger.info("As demonstrated in the example above, you can determine the reason why the team stopped by checking the {py:attr}`~autogen_agentchat.base.TaskResult.stop_reason` attribute.")

async def run_async_code_12a0d807():
    await team.reset()  # Reset the team for a new task.
    return 
 = asyncio.run(run_async_code_12a0d807())
logger.success(format_json())
async def run_async_code_0fe78289():
    await Console(team.run_stream(task="Write a short poem about the fall season."))  # Stream the messages to the console.
    return 
 = asyncio.run(run_async_code_0fe78289())
logger.success(format_json())

"""
## Resetting a Team

You can reset the team by calling the {py:meth}`~autogen_agentchat.teams.BaseGroupChat.reset` method. This method will clear the team's state, including all agents.
It will call the each agent's {py:meth}`~autogen_agentchat.base.ChatAgent.on_reset` method to clear the agent's state.
"""
logger.info("## Resetting a Team")

async def run_async_code_512f84bd():
    await team.reset()  # Reset the team for the next run.
    return 
 = asyncio.run(run_async_code_512f84bd())
logger.success(format_json())

"""
It is usually a good idea to reset the team if the next task is not related to the previous task.
However, if the next task is related to the previous task, you don't need to reset and you can instead
resume the team.

## Stopping a Team

Apart from automatic termination conditions such as {py:class}`~autogen_agentchat.conditions.TextMentionTermination`
that stops the team based on the internal state of the team, you can also stop the team
from outside by using the {py:class}`~autogen_agentchat.conditions.ExternalTermination`.

Calling {py:meth}`~autogen_agentchat.conditions.ExternalTermination.set` 
on {py:class}`~autogen_agentchat.conditions.ExternalTermination` will stop
the team when the current agent's turn is over.
Thus, the team may not stop immediately.
This allows the current agent to finish its turn and broadcast the final message to the team
before the team stops, keeping the team's state consistent.
"""
logger.info("## Stopping a Team")

external_termination = ExternalTermination()
team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=external_termination | text_termination,  # Use the bitwise OR operator to combine conditions.
)

run = asyncio.create_task(Console(team.run_stream(task="Write a short poem about the fall season.")))

async def run_async_code_80bd3181():
    await asyncio.sleep(0.1)
    return 
 = asyncio.run(run_async_code_80bd3181())
logger.success(format_json())

external_termination.set()

async def run_async_code_41cf4617():
    await run
    return 
 = asyncio.run(run_async_code_41cf4617())
logger.success(format_json())

"""
From the ouput above, you can see the team stopped because the external termination condition was met,
but the speaking agent was able to finish its turn before the team stopped.

## Resuming a Team

Teams are stateful and maintains the conversation history and context
after each run, unless you reset the team.

You can resume a team to continue from where it left off by calling the {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` or {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream` method again
without a new task.
{py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` will continue from the next agent in the round-robin order.
"""
logger.info("## Resuming a Team")

async def run_async_code_9aa6c87a():
    await Console(team.run_stream())  # Resume the team to continue the last task.
    return 
 = asyncio.run(run_async_code_9aa6c87a())
logger.success(format_json())

"""
You can see the team resumed from where it left off in the output above,
and the first message is from the next agent after the last agent that spoke
before the team stopped.

Let's resume the team again with a new task while keeping the context about the previous task.
"""
logger.info("You can see the team resumed from where it left off in the output above,")

async def run_async_code_8cb8974a():
    await Console(team.run_stream(task="将这首诗用中文唐诗风格写一遍。"))
    return 
 = asyncio.run(run_async_code_8cb8974a())
logger.success(format_json())

"""
## Aborting a Team

You can abort a call to {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` or {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream`
during execution by setting a {py:class}`~autogen_core.CancellationToken` passed to the `cancellation_token` parameter.

Different from stopping a team, aborting a team will immediately stop the team and raise a {py:class}`~asyncio.CancelledError` exception.

```{note}
The caller will get a {py:class}`~asyncio.CancelledError` exception when the team is aborted.
```
"""
logger.info("## Aborting a Team")

cancellation_token = CancellationToken()

run = asyncio.create_task(
    team.run(
        task="Translate the poem to Spanish.",
        cancellation_token=cancellation_token,
    )
)

cancellation_token.cancel()

try:
    async def run_async_code_b0a1b711():
        async def run_async_code_564a14d8():
            result = await run  # This will raise a CancelledError.
            return result
        result = asyncio.run(run_async_code_564a14d8())
        logger.success(format_json(result))
        return result
    result = asyncio.run(run_async_code_b0a1b711())
    logger.success(format_json(result))
except asyncio.CancelledError:
    logger.debug("Task was cancelled.")

"""
## Single-Agent Team

```{note}
Starting with version 0.6.2, you can use {py:class}`~autogen_agentchat.agents.AssistantAgent`
with `max_tool_iterations` to run the agent with multiple iterations
of tool calls. So you may not need to use a single-agent team if you just 
want to run the agent in a tool-calling loop.
```

Often, you may want to run a single agent in a team configuration.
This is useful for running the {py:class}`~autogen_agentchat.agents.AssistantAgent` in a loop
until a termination condition is met.

This is different from running the {py:class}`~autogen_agentchat.agents.AssistantAgent` using
its {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run` or {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run_stream` method,
which only runs the agent for one step and returns the result.
See {py:class}`~autogen_agentchat.agents.AssistantAgent` for more details about a single step.

Here is an example of running a single agent in a {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` team configuration
with a {py:class}`~autogen_agentchat.conditions.TextMessageTermination` condition.
The task is to increment a number until it reaches 10 using a tool.
The agent will keep calling the tool until the number reaches 10,
and then it will return a final {py:class}`~autogen_agentchat.messages.TextMessage`
which will stop the run.
"""
logger.info("## Single-Agent Team")


model_client = MLXChatCompletionClient(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
    parallel_tool_calls=False,  # type: ignore
)


def increment_number(number: int) -> int:
    """Increment a number by 1."""
    return number + 1


looped_assistant = AssistantAgent(
    "looped_assistant",
    model_client=model_client,
    tools=[increment_number],  # Register the tool.
    system_message="You are a helpful AI assistant, use the tool to increment the number.",
)

termination_condition = TextMessageTermination("looped_assistant")

team = RoundRobinGroupChat(
    [looped_assistant],
    termination_condition=termination_condition,
)

async for message in team.run_stream(task="Increment the number 5 to 10."):  # type: ignore
    logger.debug(type(message).__name__, message)

async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
The key is to focus on the termination condition.
In this example, we use a {py:class}`~autogen_agentchat.conditions.TextMessageTermination` condition
that stops the team when the agent stop producing {py:class}`~autogen_agentchat.messages.ToolCallSummaryMessage`.
The team will keep running until the agent produces a {py:class}`~autogen_agentchat.messages.TextMessage` with the final result.

You can also use other termination conditions to control the agent.
See [Termination Conditions](./termination.ipynb) for more details.
"""
logger.info("The key is to focus on the termination condition.")

logger.info("\n\n[DONE]", bright=True)