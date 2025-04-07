import asyncio
import json
import logging
import tempfile
from typing import Any, AsyncGenerator, List, Mapping, Sequence

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent, ToolCallSummaryMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import AgentRuntime, SingleThreadedAgentRuntime
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.logger import logger


# Simulated real-world function for task completion
def code_execution_task(input_code: str) -> str:
    """Function to simulate a task of code execution."""
    try:
        exec(input_code)
        return "Code executed successfully"
    except Exception as e:
        return f"Error executing code: {e}"


# Real-world example where an assistant and a code executor collaborate
async def task_execution_with_code(runtime: AgentRuntime | None) -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a code executor agent to run the task
        code_executor_agent = CodeExecutorAgent(
            "code_executor", code_executor=LocalCommandLineCodeExecutor(work_dir=temp_dir)
        )

        # Create an assistant agent to assist with the task
        coding_assistant_agent = AssistantAgent(
            "coding_assistant", model_client=model_client
        )

        # Setup the termination condition
        termination_condition = TextMentionTermination("TERMINATE")

        # Create a group chat with round-robin messaging pattern
        team = RoundRobinGroupChat(
            participants=[coding_assistant_agent, code_executor_agent],
            termination_condition=termination_condition,
            runtime=runtime
        )

        # Run the task with input: writing a program to print 'Hello, world!'
        result = await team.run(
            task="Write a program that prints 'Hello, world!'",
        )

        # Log the resulting messages and assert termination
        for message in result.messages:
            logger.debug(f"Message content: {message.content}")
        assert result.stop_reason == "Text 'TERMINATE' mentioned"


# Example where agents handle tools and asynchronous tasks
async def agent_with_tools_example(runtime: AgentRuntime | None) -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Define a tool to simulate a pass function for the agent
    tool = FunctionTool(code_execution_task, name="pass",
                        description="Simulate task execution")

    tool_use_agent = AssistantAgent(
        "tool_use_agent", model_client=model_client, tools=[tool])
    echo_agent = AssistantAgent("echo_agent", model_client=model_client)

    termination = TextMentionTermination("TERMINATE")

    # Create a team to execute the task using round-robin scheduling
    team = RoundRobinGroupChat(
        participants=[
            tool_use_agent, echo_agent], termination_condition=termination, runtime=runtime
    )

    result = await team.run(task="Write a program that prints 'Hello, world!'")

    # Log each message in the process
    for message in result.messages:
        logger.debug(f"Message content: {message.content}")
    assert result.stop_reason == "Text 'TERMINATE' mentioned"


# Main function to run multiple tasks
async def main():
    # Create a runtime (either single-threaded or embedded)
    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        # Run the real-world examples
        await task_execution_with_code(runtime)
        await agent_with_tools_example(runtime)
    finally:
        await runtime.stop()

# Start the execution
if __name__ == "__main__":
    asyncio.run(main())
