import asyncio
import os
import json
import logging
from typing import List, Dict, Sequence, TypedDict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.file_surfer import FileSurfer
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient

# Debug logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModuleSearchResult(TypedDict):
    module: str
    files: List[str]


class CodeSearchWorkflow:
    """Modular workflow for task-based module search in code files."""

    def __init__(
        self,
        model: str = "llama3.2",
        max_messages: int = 8
    ) -> None:
        self.model_client = OllamaChatCompletionClient(model=model)
        self.max_messages = max_messages
        self._setup_agents()
        self._setup_group_chat()

    def _setup_agents(self) -> None:
        self.task_analyzer = MagenticOneCoderAgent(
            name="TaskAnalyzer",
            description="An agent for analyzing tasks to identify necessary Python modules. Start with new tasks.",
            model_client=self.model_client,
            system_message="""
            You are a task analyzer. Analyze the user task to identify necessary REGISTERED Python modules as a comma-separated list (e.g., numpy,pandas).
            Delegate to FileSurfer for searching files.
            When complete, output JSON: {'module': ['file1.py', 'file2.py']} and end with TERMINATE.
            """
        )
        self.file_surfer = FileSurfer(
            name="FileSurfer",
            description="An agent for searching and reading .py files in a directory for module imports.",
            model_client=self.model_client,
            base_path=os.getcwd()  # Will be overridden in run
        )

    def _setup_group_chat(self) -> None:
        termination = TextMentionTermination(
            "TERMINATE") | MaxMessageTermination(max_messages=self.max_messages)
        selector_prompt = """Select an agent to perform the next step.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Start with TaskAnalyzer for new tasks. Only select FileSurfer after modules are identified.
Only select one agent.
"""
        self.group_chat = SelectorGroupChat(
            participants=[self.task_analyzer, self.file_surfer],
            model_client=self.model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,
        )

    async def run(self, task: str, directory: str) -> List[ModuleSearchResult]:
        """Run the workflow for a given task and directory."""
        # Set FileSurfer base_path to user-provided directory
        self.file_surfer._browser.base_path = os.path.abspath(directory)
        full_task = f"""
User task: {task}
Directory to search: {directory}
1. TaskAnalyzer: Identify necessary REGISTERED Python modules as a comma-separated list (e.g., numpy,pandas).
2. FileSurfer: Use open_path to navigate to {directory}, then use find_on_page_ctrl_f to search .py files for imports matching those modules (e.g., 'import module' or 'from module import'). Collect results as a list of files per module.
Output final JSON: {{'module': ['file1.py', 'file2.py']}} and end with TERMINATE.
"""
        logger.debug(f"Starting run with task: {full_task}")
        stream = self.group_chat.run_stream(task=full_task)

        # ✅ consume the async generator
        response: Response | None = None
        async for event in stream:
            response = event  # keep last yielded Response

        if response is None:
            logger.error("No response received from stream.")
            return []

        logger.debug(
            f"Received response with {len(response.messages)} messages")

        # Parse final JSON from last TextMessage
        final_content = ""
        for message in reversed(response.messages):
            logger.debug(
                f"Processing message: type={type(message).__name__}, content={getattr(message, 'content', 'None')}")
            if isinstance(message, TextMessage) and "TERMINATE" in message.content:
                content_str = message.content
                logger.debug(f"Found TERMINATE in TextMessage: {content_str}")
                if "{" in content_str and "}" in content_str:
                    try:
                        json_start = content_str.index("{")
                        json_end = content_str.rindex("}") + 1
                        final_content = content_str[json_start:json_end]
                        logger.debug(f"Extracted JSON: {final_content}")
                        break
                    except ValueError as e:
                        logger.error(
                            f"Failed to extract JSON from content: {e}")

        try:
            results: Dict[str, List[str]] = json.loads(final_content)
            logger.debug(f"Parsed results: {results}")
            return [{"module": k, "files": v} for k, v in results.items()]
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return []


# Example usage


async def example_run() -> None:
    workflow = CodeSearchWorkflow()
    results = await workflow.run(
        task="Write code agents with a teacher and memory usage",
        directory="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/autogen/examples/base"
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(example_run())
