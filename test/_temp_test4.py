from typing import List, Sequence
import asyncio
import tempfile
import os
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.replay import ReplayChatCompletionClient
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient


async def main():
    """
    Demonstrates SelectorGroupChat with a nested RoundRobinGroupChat team.
    The outer team selects between an inner team (file surfer + relevance analyzer) and a reviewer agent.
    """
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()

    # Mock model client for reproducible results
    # model_client = ReplayChatCompletionClient([
    #     "InnerTeam",
    #     "Found relevant files in /temp: ['example.py', 'utils.py']",
    #     "TERMINATE",
    #     "agent3",
    #     "Files are relevant to the task",
    #     "TERMINATE"
    # ])
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Create temporary directory for code search simulation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate some code files
        with open(os.path.join(temp_dir, "example.py"), "w") as f:
            f.write('print("Hello, world!")')
        with open(os.path.join(temp_dir, "utils.py"), "w") as f:
            f.write("def helper(): pass")

        # Create file surfer agent
        file_surfer = FileSurfer(
            name="file_surfer",
            model_client=model_client,
            description="Navigates and reads code files in a directory",
            base_path=temp_dir
        )

        # Create relevance analyzer agent
        relevance_analyzer = AssistantAgent(
            name="relevance_analyzer",
            model_client=model_client,
            description="Analyzes code relevance",
            system_message="You analyze the relevance of found code files to the task."
        )

        # Create inner team
        inner_team = RoundRobinGroupChat(
            participants=[file_surfer, relevance_analyzer],
            termination_condition=TextMentionTermination("TERMINATE"),
            runtime=runtime,
            name="InnerTeam",
            description="Team that searches and analyzes code relevance"
        )

        # Create reviewer agent
        reviewer = AssistantAgent(
            name="agent3",
            model_client=model_client,
            description="Reviews search results",
            system_message="You review the relevance of found code files."
        )

        # Create outer team
        outer_team = SelectorGroupChat(
            participants=[inner_team, reviewer],
            model_client=model_client,
            termination_condition=TextMentionTermination("TERMINATE"),
            runtime=runtime
        )

        # Run the task
        task = f"Find Python code files in {temp_dir} relevant to printing a greeting message"
        result = await outer_team.run(task=task)

        # Print results
        print("Task Result:")
        for message in result.messages:
            print(f"{message.source}: {message.content}")
        print(f"Stop Reason: {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())
