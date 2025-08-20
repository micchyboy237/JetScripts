import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
from typing import List
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
# Mixture of Agents

[Mixture of Agents](https://arxiv.org/abs/2406.04692) is a multi-agent design pattern
that models after the feed-forward neural network architecture.

The pattern consists of two types of agents: worker agents and a single orchestrator agent.
Worker agents are organized into multiple layers, with each layer consisting of a fixed number of worker agents.
Messages from the worker agents in a previous layer are concatenated and sent to
all the worker agents in the next layer.

This example implements the Mixture of Agents pattern using the core library
following the [original implementation](https://github.com/togethercomputer/moa) of multi-layer mixture of agents.

Here is a high-level procedure overview of the pattern:
1. The orchestrator agent takes input a user task and first dispatches it to the worker agents in the first layer.
2. The worker agents in the first layer process the task and return the results to the orchestrator agent.
3. The orchestrator agent then synthesizes the results from the first layer and dispatches an updated task with the previous results to the worker agents in the second layer.
4. The process continues until the final layer is reached.
5. In the final layer, the orchestrator agent aggregates the results from previous layer and returns a single final result to the user.

We use the direct messaging API {py:meth}`~autogen_core.BaseAgent.send_message` to implement this pattern.
This makes it easier to add more features like worker task cancellation and error handling in the future.
"""
logger.info("# Mixture of Agents")



"""
## Message Protocol

The agents communicate using the following messages:
"""
logger.info("## Message Protocol")

@dataclass
class WorkerTask:
    task: str
    previous_results: List[str]


@dataclass
class WorkerTaskResult:
    result: str


@dataclass
class UserTask:
    task: str


@dataclass
class FinalResult:
    result: str

"""
## Worker Agent

Each worker agent receives a task from the orchestrator agent and processes them
indepedently.
Once the task is completed, the worker agent returns the result.
"""
logger.info("## Worker Agent")

class WorkerAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(description="Worker Agent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: WorkerTask, ctx: MessageContext) -> WorkerTaskResult:
        if message.previous_results:
            system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
            system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(message.previous_results)])
            model_result = await self._model_client.create(
                [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
            )
        else:
            model_result = await self._model_client.create([UserMessage(content=message.task, source="user")])
        assert isinstance(model_result.content, str)
        logger.debug(f"{'-'*80}\nWorker-{self.id}:\n{model_result.content}")
        return WorkerTaskResult(result=model_result.content)

"""
## Orchestrator Agent

The orchestrator agent receives tasks from the user and distributes them to the worker agents,
iterating over multiple layers of worker agents. Once all worker agents have processed the task,
the orchestrator agent aggregates the results and publishes the final result.
"""
logger.info("## Orchestrator Agent")

class OrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
    ) -> None:
        super().__init__(description="Aggregator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> FinalResult:
        logger.debug(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived task: {message.task}")
        worker_task = WorkerTask(task=message.task, previous_results=[])
        for i in range(self._num_layers - 1):
            worker_ids = [
                AgentId(worker_type, f"{self.id.key}/layer_{i}/worker_{j}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            logger.debug(f"{'-'*80}\nOrchestrator-{self.id}:\nDispatch to workers at layer {i}")
            results = await asyncio.gather(*[self.send_message(worker_task, worker_id) for worker_id in worker_ids])
            logger.debug(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived results from workers at layer {i}")
            worker_task = WorkerTask(task=message.task, previous_results=[r.result for r in results])
        logger.debug(f"{'-'*80}\nOrchestrator-{self.id}:\nPerforming final aggregation")
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(worker_task.previous_results)])
        model_result = await self._model_client.create(
            [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)

"""
## Running Mixture of Agents

Let's run the mixture of agents on a math task. You can change the task to make it more challenging, for example, by trying tasks from the [International Mathematical Olympiad](https://www.imo-official.org/problems.aspx).
"""
logger.info("## Running Mixture of Agents")

task = (
    "I have 432 cookies, and divide them 3:4:2 between Alice, Bob, and Charlie. How many cookies does each person get?"
)

"""
Let's set up the runtime with 3 layers of worker agents, each layer consisting of 3 worker agents.
We only need to register a single worker agent types, "worker", because we are using
the same model client configuration (i.e., qwen3-1.7b-4bit-mini) for all worker agents.
If you want to use different models, you will need to register multiple worker agent types,
one for each model, and update the `worker_agent_types` list in the orchestrator agent's
factory function.

The instances of worker agents are automatically created when the orchestrator agent
dispatches tasks to them.
See [Agent Identity and Lifecycle](../core-concepts/agent-identity-and-lifecycle.md)
for more information on agent lifecycle.
"""
logger.info("Let's set up the runtime with 3 layers of worker agents, each layer consisting of 3 worker agents.")

runtime = SingleThreadedAgentRuntime()
model_client = MLXAutogenChatLLMAdapter(model="qwen3-1.7b-4bit-mini")
async def run_async_code_a3cbf129():
    await WorkerAgent.register(runtime, "worker", lambda: WorkerAgent(model_client=model_client))
asyncio.run(run_async_code_a3cbf129())
async def async_func_3():
    await OrchestratorAgent.register(
        runtime,
        "orchestrator",
        lambda: OrchestratorAgent(model_client=model_client, worker_agent_types=["worker"] * 3, num_layers=3),
    )
asyncio.run(async_func_3())

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_5d35fbb3():
    result = await runtime.send_message(UserTask(task=task), AgentId("orchestrator", "default"))
    return result
result = asyncio.run(run_async_code_5d35fbb3())
logger.success(format_json(result))

async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())
async def run_async_code_0349fda4():
    await model_client.close()
asyncio.run(run_async_code_0349fda4())

logger.debug(f"{'-'*80}\nFinal result:\n{result.result}")

logger.info("\n\n[DONE]", bright=True)