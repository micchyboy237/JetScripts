import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Image
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import List
import os
import re
import shutil
import tempfile


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Code Execution

In this section we explore creating custom agents to handle code generation and execution. These tasks can be handled using the provided Agent implementations found here {py:meth}`~autogen_agentchat.agents.AssistantAgent`, {py:meth}`~autogen_agentchat.agents.CodeExecutorAgent`; but this guide will show you how to implement custom, lightweight agents that can replace their functionality. This simple example implements two agents that create a plot of Tesla's and Nvidia's stock returns.

We first define the agent classes and their respective procedures for
handling messages.
We create two agent classes: `Assistant` and `Executor`. The `Assistant`
agent writes code and the `Executor` agent executes the code.
We also create a `Message` data class, which defines the messages that are passed between
the agents.

```{attention}
Code generated in this example is run within a [Docker](https://www.docker.com/) container. Please ensure Docker is [installed](https://docs.docker.com/get-started/get-docker/) and running prior to running the example. Local code execution is available ({py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`) but is not recommended due to the risk of running LLM generated code in your local environment.
```
"""
logger.info("# Code Execution")


@dataclass
class Message:
    content: str


@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""Write Python script in markdown block, and it will be executed.
Always save figures to file in the current directory. Do not use plt.show(). All code required to complete this task must be contained within a single response.""",
            )
        ]

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        self._chat_history.append(UserMessage(
            content=message.content, source="user"))

        result = await self._model_client.create(self._chat_history)
        logger.success(format_json(result))
        logger.debug(f"\n{'-'*80}\nAssistant:\n{result.content}")
        self._chat_history.append(AssistantMessage(
            content=result.content, source="assistant"))  # type: ignore

        await self.publish_message(Message(content=result.content), DefaultTopicId())
        logger.success(format_json(result))


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        code_blocks = extract_markdown_code_blocks(message.content)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            logger.success(format_json(result))
            logger.debug(f"\n{'-'*80}\nExecutor:\n{result.output}")
            await self.publish_message(Message(content=result.output), DefaultTopicId())
            logger.success(format_json(result))


"""
You might have already noticed, the agents' logic, whether it is using model or code executor,
is completely decoupled from
how messages are delivered. This is the core idea: the framework provides
a communication infrastructure, and the agents are responsible for their own
logic. We call the communication infrastructure an **Agent Runtime**.

Agent runtime is a key concept of this framework. Besides delivering messages,
it also manages agents' lifecycle. 
So the creation of agents are handled by the runtime.

The following code shows how to register and run the agents using 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
a local embedded agent runtime implementation.
"""
logger.info(
    "You might have already noticed, the agents' logic, whether it is using model or code executor,")

# Set DOCKER_HOST to Colima's socket
os.environ["DOCKER_HOST"] = "unix:///Users/jethroestrada/.colima/default/docker.sock"

work_dir = tempfile.mkdtemp()

runtime = SingleThreadedAgentRuntime()


async def async_func_10():
    # type: ignore[syntax]
    async with DockerCommandLineCodeExecutor(
        work_dir=work_dir,
        bind_dir=str(work_dir),
        timeout=60,
        auto_remove=False,
        # delete_tmp_files=True
    ) as executor:
        model_client = OllamaChatCompletionClient(
            model="llama3.2",
            host="http://localhost:11434"
        )
        await Assistant.register(
            runtime,
            "assistant",
            lambda: Assistant(model_client=model_client),
        )
        await Executor.register(runtime, "executor", lambda: Executor(executor))

        runtime.start()
        await runtime.publish_message(
            Message(
                "Create a plot of NVIDA vs TSLA stock returns YTD from 2024-01-01."), DefaultTopicId()
        )

        await runtime.stop_when_idle()
        await model_client.close()
    return result

result = asyncio.run(async_func_10())
logger.success(format_json(result))

"""
From the agent's output, we can see the plot of Tesla's and Nvidia's stock returns
has been created.
"""
logger.info(
    "From the agent's output, we can see the plot of Tesla's and Nvidia's stock returns")


Image(filename=f"{work_dir}/nvidia_vs_tesla_ytd_returns.png")  # type: ignore

"""
AutoGen also supports a distributed agent runtime, which can host agents running on
different processes or machines, with different identities, languages and dependencies.

To learn how to use agent runtime, communication, message handling, and subscription, please continue
reading the sections following this quick start.
"""
logger.info(
    "AutoGen also supports a distributed agent runtime, which can host agents running on")

logger.info("\n\n[DONE]", bright=True)
