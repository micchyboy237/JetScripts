import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
DefaultTopicId,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
TypeSubscription,
default_subscription,
message_handler,
)
from autogen_core.models import (
AssistantMessage,
ChatCompletionClient,
LLMMessage,
SystemMessage,
UserMessage,
)
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
from typing import Dict, List
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Multi-Agent Debate

Multi-Agent Debate is a multi-agent design pattern that simulates a multi-turn interaction
where in each turn, agents exchange their responses with each other, and refine
their responses based on the responses from other agents.

This example shows an implementation of the multi-agent debate pattern for solving
math problems from the [GSM8K benchmark](https://huggingface.co/datasets/openai/gsm8k).

There are of two types of agents in this pattern: solver agents and an aggregator agent.
The solver agents are connected in a sparse manner following the technique described in
[Improving Multi-Agent Debate with Sparse Communication Topology](https://arxiv.org/abs/2406.11776).
The solver agents are responsible for solving math problems and exchanging responses with each other.
The aggregator agent is responsible for distributing math problems to the solver agents,
waiting for their final responses, and aggregating the responses to get the final answer.

The pattern works as follows:
1. User sends a math problem to the aggregator agent.
2. The aggregator agent distributes the problem to the solver agents.
3. Each solver agent processes the problem, and publishes a response to its neighbors.
4. Each solver agent uses the responses from its neighbors to refine its response, and publishes a new response.
5. Repeat step 4 for a fixed number of rounds. In the final round, each solver agent publishes a final response.
6. The aggregator agent uses majority voting to aggregate the final responses from all solver agents to get a final answer, and publishes the answer.

We will be using the broadcast API, i.e., {py:meth}`~autogen_core.BaseAgent.publish_message`,
and we will be using topic and subscription to implement the communication topology.
Read about [Topics and Subscriptions](../core-concepts/topic-and-subscription.md) to understand how they work.
"""
logger.info("# Multi-Agent Debate")


"""
## Message Protocol

First, we define the messages used by the agents.
`IntermediateSolverResponse` is the message exchanged among the solver agents in each round,
and `FinalSolverResponse` is the message published by the solver agents in the final round.
"""
logger.info("## Message Protocol")


@dataclass
class Question:
    content: str


@dataclass
class Answer:
    content: str


@dataclass
class SolverRequest:
    content: str
    question: str


@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int


@dataclass
class FinalSolverResponse:
    answer: str


"""
## Solver Agent

The solver agent is responsible for solving math problems and exchanging responses with other solver agents.
Upon receiving a `SolverRequest`, the solver agent uses an LLM to generate an answer.
Then, it publishes a `IntermediateSolverResponse`
or a `FinalSolverResponse` based on the round number.

The solver agent is given a topic type, which is used to indicate the topic
to which the agent should publish intermediate responses. This topic is subscribed
to by its neighbors to receive responses from this agent -- we will show
how this is done later.

We use {py:meth}`~autogen_core.components.default_subscription` to let
solver agents subscribe to the default topic, which is used by the aggregator agent
to collect the final responses from the solver agents.
"""
logger.info("## Solver Agent")


@default_subscription
class MathSolver(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, topic_type: str, num_neighbors: int, max_round: int) -> None:
        super().__init__("A debator.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateSolverResponse]] = {}
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant with expertise in mathematics and reasoning. "
                    "Your task is to assist in solving a math reasoning problem by providing "
                    "a clear and detailed solution. Limit your output within 100 words, "
                    "and your final answer should be a single numerical number, "
                    "in the form of {{answer}}, at the end of your response. "
                    "For example, 'The answer is {{42}}.'"
                )
            )
        ]
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx: MessageContext) -> None:
        self._history.append(UserMessage(
            content=message.content, source="user"))

        async def run_async_code_27958a41():
            async def run_async_code_3135ca92():
                model_result = await self._model_client.create(self._system_messages + self._history)
                return model_result
            model_result = asyncio.run(run_async_code_3135ca92())
            logger.success(format_json(model_result))
            return model_result
        model_result = asyncio.run(run_async_code_27958a41())
        logger.success(format_json(model_result))
        assert isinstance(model_result.content, str)
        self._history.append(AssistantMessage(
            content=model_result.content, source=self.metadata["type"]))
        logger.debug(
            f"{'-'*80}\nSolver {self.id} round {self._round}:\n{model_result.content}")
        match = re.search(r"\{\{(\-?\d+(\.\d+)?)\}\}", model_result.content)
        if match is None:
            raise ValueError("The model response does not contain the answer.")
        answer = match.group(1)
        self._round += 1
        if self._round == self._max_round:
            async def run_async_code_1bfbe7e4():
                await self.publish_message(FinalSolverResponse(answer=answer), topic_id=DefaultTopicId())
                return
             = asyncio.run(run_async_code_1bfbe7e4())
            logger.success(format_json())
        else:
            await self.publish_message(
                IntermediateSolverResponse(
                    content=model_result.content,
                    question=message.question,
                    answer=answer,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(type=self._topic_type),
            )

    @message_handler
    async def handle_response(self, message: IntermediateSolverResponse, ctx: MessageContext) -> None:
        self._buffer.setdefault(message.round, []).append(message)
        if len(self._buffer[message.round]) == self._num_neighbors:
            logger.debug(
                f"{'-'*80}\nSolver {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors."
            )
            prompt = "These are the solutions to the problem from other agents:\n"
            for resp in self._buffer[message.round]:
                prompt += f"One agent solution: {resp.content}\n"
            prompt += (
                "Using the solutions from other agents as additional information, "
                "can you provide your answer to the math problem? "
                f"The original math problem is {message.question}. "
                "Your final answer should be a single numerical number, "
                "in the form of {{answer}}, at the end of your response."
            )
            async def run_async_code_63d77071():
                await self.send_message(SolverRequest(content=prompt, question=message.question), self.id)
                return 
             = asyncio.run(run_async_code_63d77071())
            logger.success(format_json())
            self._buffer.pop(message.round)

"""
## Aggregator Agent

The aggregator agent is responsible for handling user question and 
distributing math problems to the solver agents.

The aggregator subscribes to the default topic using
{py:meth}`~autogen_core.components.default_subscription`. The default topic is used to
recieve user question, receive the final responses from the solver agents,
and publish the final answer back to the user.

In a more complex application when you want to isolate the multi-agent debate into a
sub-component, you should use
{py:meth}`~autogen_core.components.type_subscription` to set a specific topic
type for the aggregator-solver communication, 
and have the both the solver and aggregator publish and subscribe to that topic type.
"""
logger.info("## Aggregator Agent")

@default_subscription
class MathAggregator(RoutedAgent):
    def __init__(self, num_solvers: int) -> None:
        super().__init__("Math Aggregator")
        self._num_solvers = num_solvers
        self._buffer: List[FinalSolverResponse] = []

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        logger.debug(f"{'-'*80}\nAggregator {self.id} received question:\n{message.content}")
        prompt = (
            f"Can you solve the following math problem?\n{message.content}\n"
            "Explain your reasoning. Your final answer should be a single numerical number, "
            "in the form of {{answer}}, at the end of your response."
        )
        logger.debug(f"{'-'*80}\nAggregator {self.id} publishes initial solver request.")
        async def run_async_code_c766de2c():
            await self.publish_message(SolverRequest(content=prompt, question=message.content), topic_id=DefaultTopicId())
            return 
         = asyncio.run(run_async_code_c766de2c())
        logger.success(format_json())

    @message_handler
    async def handle_final_solver_response(self, message: FinalSolverResponse, ctx: MessageContext) -> None:
        self._buffer.append(message)
        if len(self._buffer) == self._num_solvers:
            logger.debug(f"{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            answers = [resp.answer for resp in self._buffer]
            majority_answer = max(set(answers), key=answers.count)
            async def run_async_code_e2f5f630():
                await self.publish_message(Answer(content=majority_answer), topic_id=DefaultTopicId())
                return 
             = asyncio.run(run_async_code_e2f5f630())
            logger.success(format_json())
            self._buffer.clear()
            logger.debug(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}")

"""
## Setting Up a Debate

We will now set up a multi-agent debate with 4 solver agents and 1 aggregator agent.
The solver agents will be connected in a sparse manner as illustrated in the figure
below:

```
A --- B
|     |
|     |
D --- C
```

Each solver agent is connected to two other solver agents. 
For example, agent A is connected to agents B and C.

Let's first create a runtime and register the agent types.
"""
logger.info("## Setting Up a Debate")

runtime = SingleThreadedAgentRuntime()

model_client = MLXAutogenChatLLMAdapter(model="llama-3.2-3b-instruct")

await MathSolver.register(
    runtime,
    "MathSolverA",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverA",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverB",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverB",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverC",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverC",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverD",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverD",
        num_neighbors=2,
        max_round=3,
    ),
)
async def run_async_code_24b0fde2():
    await MathAggregator.register(runtime, "MathAggregator", lambda: MathAggregator(num_solvers=4))
    return 
 = asyncio.run(run_async_code_24b0fde2())
logger.success(format_json())

"""
Now we will create the solver agent topology using {py:class}`~autogen_core.components.TypeSubscription`,
which maps each solver agent's publishing topic type to its neighbors' agent types.
"""
logger.info("Now we will create the solver agent topology using {py:class}`~autogen_core.components.TypeSubscription`,")

async def run_async_code_67a46a42():
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverD"))
    return 
 = asyncio.run(run_async_code_67a46a42())
logger.success(format_json())
async def run_async_code_d1698d92():
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverB"))
    return 
 = asyncio.run(run_async_code_d1698d92())
logger.success(format_json())

async def run_async_code_cf3fff1c():
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverA"))
    return 
 = asyncio.run(run_async_code_cf3fff1c())
logger.success(format_json())
async def run_async_code_08105e28():
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverC"))
    return 
 = asyncio.run(run_async_code_08105e28())
logger.success(format_json())

async def run_async_code_2526e699():
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverB"))
    return 
 = asyncio.run(run_async_code_2526e699())
logger.success(format_json())
async def run_async_code_103bf518():
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverD"))
    return 
 = asyncio.run(run_async_code_103bf518())
logger.success(format_json())

async def run_async_code_0906cda7():
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverC"))
    return 
 = asyncio.run(run_async_code_0906cda7())
logger.success(format_json())
async def run_async_code_ab7cbd64():
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverA"))
    return 
 = asyncio.run(run_async_code_ab7cbd64())
logger.success(format_json())

"""
## Solving Math Problems

Now let's run the debate to solve a math problem.
We publish a `SolverRequest` to the default topic, 
and the aggregator agent will start the debate.
"""
logger.info("## Solving Math Problems")

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
runtime.start()
async def run_async_code_33e3430c():
    await runtime.publish_message(Question(content=question), DefaultTopicId())
    return 
 = asyncio.run(run_async_code_33e3430c())
logger.success(format_json())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
    return 
 = asyncio.run(run_async_code_b7ca34d4())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)