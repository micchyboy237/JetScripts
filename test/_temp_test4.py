import asyncio
import os
from typing import List, Dict, Sequence, TypedDict, Literal

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, UserMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient


class ModuleSearchResult(TypedDict):
    module: str
    files: List[str]


class CodeSearchWorkflow:
    """Modular workflow for task-based module search in code files."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = os.environ.get("OPENAI_API_KEY", ""),
        work_dir: str = "temp_code_dir",
        max_messages: int = 20
    ) -> None:
        self.model_client = OpenAIChatCompletionClient(
            model=model, api_key=api_key)
        self.code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
        self.max_messages = max_messages
        self._setup_agents()
        self._setup_group_chat()

    def _setup_agents(self) -> None:
        self.task_analyzer = AssistantAgent(
            name="TaskAnalyzer",
            model_client=self.model_client,
            system_message=(
                "Analyze the task and output a comma-separated list of necessary Python modules. "
                "Example output: numpy,pandas,sklearn"
            )
        )
        self.code_searcher = CodeExecutorAgent(
            name="CodeSearcher",
            model_client=self.model_client,
            code_executor=self.code_executor,
            system_message=(
                "Given modules and a directory, write and execute Python code to search .py files "
                "for imports matching those modules (use re for matching 'import module' or 'from module'). "
                "Return results as JSON: {'module': ['file1.py', 'file2.py']}. Handle errors gracefully."
            )
        )

    def _setup_group_chat(self) -> None:
        self.group_chat = RoundRobinGroupChat(
            agents=[self.task_analyzer, self.code_searcher],
            condition=MaxMessageTermination(max_messages=self.max_messages)
        )

    async def run(self, task: str, directory: str) -> List[ModuleSearchResult]:
        """Run the workflow for a given task and directory."""
        prompt = f"Task: {task}\nDirectory to search: {directory}"
        initial_messages: Sequence[BaseChatMessage] = [
            UserMessage(content=prompt)]
        response: Response = await self.group_chat.run(
            messages=initial_messages,
            cancellation_token=CancellationToken()
        )
        # Extract final result from response (parse JSON from last message)
        final_content = response.messages[-1].content if response.messages else ""
        try:
            # Simplified; use json.loads in prod
            results: Dict[str, List[str]] = eval(final_content)
            return [{"module": k, "files": v} for k, v in results.items()]
        except Exception:
            return []

# Example usage


async def example_run() -> None:
    workflow = CodeSearchWorkflow(api_key="your_api_key_here")
    results = await workflow.run(
        task="build a machine learning model for image classification",
        directory="/path/to/your/codebase"
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(example_run())
