import asyncio
import json
import os
from typing import List, Dict, Sequence, TypedDict, Literal

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage, BaseAgentEvent, BaseChatMessage, HandoffMessage, MultiModalMessage, StopMessage
from autogen_agentchat.base import Response, TerminationCondition, TerminatedException
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.utils import content_to_str
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage, LLMMessage
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient


class ModuleSearchResult(TypedDict):
    module: str
    files: List[str]


class LLMTermination(TerminationCondition):
    """Terminate the conversation if an LLM determines the task is complete.

    Args:
        prompt: The prompt to evaluate in the llm
        model_client: The LLM model_client to use
        termination_phrase: The phrase to look for in the LLM output to trigger termination
    """

    def __init__(self, prompt: str, model_client: ChatCompletionClient, termination_phrase: str = "TERMINATE") -> None:
        self._prompt = prompt
        self._model_client = model_client
        self._termination_phrase = termination_phrase
        self._terminated = False
        self._context: Sequence[LLMMessage] = []

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException(
                "Termination condition has already been reached")

        # Build the context
        for message in messages:
            if isinstance(message, TextMessage):
                self._context.append(UserMessage(
                    content=message.content, source=message.source))
            elif isinstance(message, MultiModalMessage):
                if self._model_client.model_info.get("vision", False):
                    self._context.append(UserMessage(
                        content=message.content, source=message.source))
                else:
                    self._context.append(UserMessage(content=content_to_str(
                        message.content), source=message.source))

        if len(self._context) == 0:
            return None

        # Call the model to evaluate
        response = await self._model_client.create(self._context + [UserMessage(content=self._prompt, source="user")])

        # Check for termination
        if isinstance(response.content, str) and self._termination_phrase in response.content:
            self._terminated = True
            return StopMessage(content=response.content, source="LLMTermination")
        return None

    async def reset(self) -> None:
        self._terminated = False
        self._context = []


class CodeSearchWorkflow:
    """Modular workflow for task-based module search in code files."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = os.environ.get("OPENAI_API_KEY", ""),
        max_messages: int = 20
    ) -> None:
        self.model_client = OllamaChatCompletionClient(
            model=model, api_key=api_key)
        self.max_messages = max_messages
        self._setup_agents()
        self._setup_group_chat()

    def _setup_agents(self) -> None:
        self.task_analyzer = MagenticOneCoderAgent(
            name="TaskAnalyzer",
            model_client=self.model_client,
        )
        self.file_surfer = FileSurfer(
            name="FileSurfer",
            model_client=self.model_client,
        )

    def _setup_group_chat(self) -> None:
        termination_prompt = """Consider the following task:
Analyze the user task to identify necessary REGISTERED Python modules as a comma-separated list, then search .py files in the specified directory for imports matching those modules, returning results as JSON: {'module': ['file1.py', 'file2.py']}.

Does the above conversation suggest that the task has been solved?
If so, reply "TERMINATE", otherwise reply "CONTINUE"
"""
        llm_termination = LLMTermination(
            prompt=termination_prompt,
            model_client=self.model_client
        )
        termination = MaxMessageTermination(
            max_messages=self.max_messages) | llm_termination
        self.group_chat = SelectorGroupChat(
            agents=[self.task_analyzer, self.file_surfer],
            model_client=self.model_client,
            termination_condition=termination
        )

    async def run(self, task: str, directory: str) -> List[ModuleSearchResult]:
        """Run the workflow for a given task and directory."""
        prompt = f"User task: {task}\nDirectory to search: {directory}\nAnalyze to identify necessary Python modules, then search .py files in {directory} for those modules. Return final JSON: {{'module': ['file1.py', 'file2.py']}}"
        initial_messages: Sequence[BaseChatMessage] = [
            UserMessage(content=prompt)]
        response: Response = await self.group_chat.run(
            messages=initial_messages,
            cancellation_token=CancellationToken()
        )
        # Final summarization
        final_context: Sequence[LLMMessage] = []
        for message in response.messages:
            if isinstance(message, TextMessage):
                final_context.append(UserMessage(
                    content=message.content, source=message.source))
            elif isinstance(message, MultiModalMessage):
                final_context.append(UserMessage(content=content_to_str(
                    message.content), source=message.source))
        final_context.append(UserMessage(
            content="Extract the final search results from the conversation as JSON dict of module to list of files."
        ))
        final_response = await self.model_client.create(final_context)
        if isinstance(final_response.content, dict):
            final_content = json.dumps(final_response.content)
        else:
            final_content = final_response.content if isinstance(
                final_response.content, str) else ""
        try:
            results: Dict[str, List[str]] = json.loads(final_content)
            return [{"module": k, "files": v} for k, v in results.items()]
        except Exception:
            return []


async def example_run() -> None:
    workflow = CodeSearchWorkflow(api_key="your_api_key_here")
    results = await workflow.run(
        task="Write code agents with a teacher and memory usage",
        directory="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/autogen/examples/base"
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(example_run())
