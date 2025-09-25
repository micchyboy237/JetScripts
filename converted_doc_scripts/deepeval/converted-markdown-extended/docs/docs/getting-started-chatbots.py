from jet.transformers.formatters import format_json
from agents import Agent, Runner, SQLiteSession
from datetime import datetime
from deepeval import evaluate
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.dataset import EvaluationDataset, ConversationalGolden
from deepeval.metrics import TurnRelevancyMetric
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case import Turn
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.llama_index.ollama_function_calling import Ollama
from jet.logger import logger
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from ollama import Ollama
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from typing import List
import CodeBlock from "@theme/CodeBlock";
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import os
import shutil
import { Timeline, TimelineItem } from "@site/src/components/Timeline";


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)


"""
---
id: getting-started-chatbots
title: Chatbot Evaluation
sidebar_label: Chatbots
---


Learn to evaluate any multi-turn chatbot using `deepeval` - including QA agents, customer support chatbots, and even chatrooms.

## Overview

Chatbot Evaluation is different from other types of evaluations because unlike single-turn tasks, conversations happen over multiple "turns". This means your chatbot must stay context-aware across the conversation, and not just accurate in individual responses.

**In this 10 min quickstart, you'll learn how to:**

- Prepare conversational test cases
- Evaluate chatbot conversations
- Simulate users interactions

## Prerequisites

- Install `deepeval`
- A Confident AI API key (recommended). Sign up for one [here.](https://app.confident-ai.com)

:::info
Confident AI allows you to view and share your chatbot testing reports. Set your API key in the CLI:
"""
logger.info("## Overview")

CONFIDENT_API_KEY="confident_us..."

"""
:::

## Understanding Multi-Turn Evals

Multi-turn evals are tricky because of the ad-hoc nature of conversations. The nth AI output will depend on the (n-1)th user input, and this depends on all prior turns up until the initial message.

Hence, when running evals for the purpose of benchmarking we cannot compare different conversations by looking at their turns. In `deepeval`, multi-turn interactions are grouped by **scenarios** instead. If two conversations occur under the same scenario, we consider those the same.

![Conversational Test Case](https://deepeval-docs.s3.amazonaws.com/docs:conversational-test-case.png)

:::note
Scenarios are optional in the diagram because not all users start with conversations with labelled scenarios.
:::

## Run A Multi-Turn Eval

In `deepeval`, chatbots are evaluated as multi-turn **interactions**. In code, you'll have to format them into test cases, which adheres to Ollama's messages format.

<Timeline>
<TimelineItem title="Create a test case">

Create a `ConversationalTestCase` by passing in a list of `Turn`s from an existing conversation, similar to Ollama's message format.
"""
logger.info("## Understanding Multi-Turn Evals")


test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hello, how are you?"),
        Turn(role="assistant", content="I'm doing well, thank you!"),
        Turn(role="user", content="How can I help you today?"),
        Turn(role="assistant", content="I'd like to buy a ticket to a Coldplay concert."),
    ]
)

"""
You can learn about a `Turn`'s data model [here.](/docs/evaluation-multiturn-test-cases#turns)

</TimelineItem>
<TimelineItem title="Run an evaluation">

Run an evaluation on the test case using `deepeval`'s multi-turn metrics, or create your own using [Conversational G-Eval](/docs/metrics-conversational-g-eval).
"""
logger.info("You can learn about a `Turn`'s data model [here.](/docs/evaluation-multiturn-test-cases#turns)")

...

evaluate(test_cases=[test_case], metrics=[TurnRelevancyMetric(), KnowledgeRetentionMetric()])

"""
Finally run `main.py`:
"""
logger.info("Finally run `main.py`:")

python main.py

"""
🎉🥳 **Congratulations!** You've just ran your first multi-turn eval. Here's what happened:

- When you call `evaluate()`, `deepeval` runs all your `metrics` against all `test_cases`
- All `metrics` outputs a score between `0-1`, with a `threshold` defaulted to `0.5`
- A test case passes only if all metrics passess

This creates a test run, which is a "snapshot"/benchmark of your multi-turn chatbot at any point in time.

</TimelineItem>
<TimelineItem title="View on Confident AI (recommended)">

If you've set your `CONFIDENT_API_KEY`, test runs will appear automatically on [Confident AI](https://app.confident-ai.com), the DeepEval platform.

<VideoDisplayer src="https://deepeval-docs.s3.us-east-1.amazonaws.com/getting-started%3Aconversation-test-report.mp4" />

:::tip
If you haven't logged in, you can still upload the test run to Confident AI from local cache:
"""
logger.info("This creates a test run, which is a "snapshot"/benchmark of your multi-turn chatbot at any point in time.")

deepeval view

"""
:::

</TimelineItem>
</Timeline>

## Working With Datasets

Although we ran an evaluation in the previous section, it's not very useful because it is far from a standardized benchmark. To create a standardized benchmark for evals, use `deepeval`'s datasets:
"""
logger.info("## Working With Datasets")


dataset = EvaluationDataset(
  goldens=[
    ConversationalGolden(scenario="Angry user asking for a refund"),
    ConversationalGolden(scenario="Couple booking two VIP Coldplay tickets")
  ]
)

"""
A dataset is a collection of goldens in `deepeval`, and in a multi-turn context this these are represented by `ConversationalGolden`s.

![Evaluation Dataset](https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:evaluation-dataset.png)

The idea is simple - we start with a list of standardized `scenario`s for each golden, and we'll simulate turns during evaluation time for more robust evaluation.

## Simulate Turns for Evals

Evaluating your chatbot from [simulated turns](/docs/getting-started-chatbots#evaluate-chatbots-from-simulations) is **the best** approach for multi-turn evals, because it:

- Standardizes your test bench, unlike ad-hoc evals
- Automates the process of manual prompting, which can take hours

Both of which are solved using `deepeval`'s `ConversationSimulator`.

<Timeline>
<TimelineItem title="Create dataset of goldens">

Create a `ConversationalGolden` by providing your user description, scenario, and expected outcome, for the conversation you wish to simulate.
"""
logger.info("## Simulate Turns for Evals")


golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)

dataset = EvaluationDataset(goldens=[golden])

"""
If you've set your `CONFIDENT_API_KEY` correctly, you can save them on the platform to collaborate with your team:
"""
logger.info("If you've set your `CONFIDENT_API_KEY` correctly, you can save them on the platform to collaborate with your team:")

dataset.push(alias="A new multi-turn dataset")

"""
<VideoDisplayer src="http://deepeval-docs.s3.us-east-1.amazonaws.com/getting-started%3Achatbot-evals%3Amultiturn-dataset.mp4" />

</TimelineItem>
<TimelineItem title="Wrap chatbot in callback">

Define a callback function to generate the **next chatbot response** in a conversation, given the conversation history.

<Tabs groupId="techstack">
<TabItem value="python" label="Python">
"""
logger.info("Define a callback function to generate the **next chatbot response** in a conversation, given the conversation history.")


async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    response = await your_chatbot(input, turns, thread_id)
    logger.success(format_json(response))
    return Turn(role="assistant", content=response)

"""
</TabItem>
<TabItem value="ollama" label="Ollama">
"""


client = Ollama()

async def model_callback(input: str, turns: List[Turn]) -> str:
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(model="llama3.2", messages=messages)
    logger.success(format_json(response))
    return Turn(role="assistant", content=response.choices[0].message.content)

"""
</TabItem>
<TabItem value="langchain" label="LangChain">
"""


store = {}
llm = ChatOllama(model="llama3.2")
prompt = ChatPromptTemplate.from_messages([("system", "You are a ticket purchasing assistant."), MessagesPlaceholder(variable_name="history"), ("human", "{input}")])
chain_with_history = RunnableWithMessageHistory(prompt | llm, lambda session_id: store.setdefault(session_id, ChatMessageHistory()), input_messages_key="input", history_messages_key="history")

async def model_callback(input: str, thread_id: str) -> Turn:
    response = chain_with_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": thread_id}}
    )
    return Turn(role="assistant", content=response.content)

"""
</TabItem>
<TabItem value="llama_index" label="LlamaIndex">
"""


chat_store = SimpleChatStore()
llm = Ollama(model="llama3.2")

async def model_callback(input: str, thread_id: str) -> Turn:
    memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=thread_id)
    chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
    response = chat_engine.chat(input)
    return Turn(role="assistant", content=response.response)

"""
</TabItem>
<TabItem value="ollama-agents" label="Ollama Agents">
"""


sessions = {}
agent = Agent(name="Test Assistant", instructions="You are a helpful assistant that answers questions concisely.")

async def model_callback(input: str, thread_id: str) -> Turn:
    if thread_id not in sessions:
        sessions[thread_id] = SQLiteSession(thread_id)
    session = sessions[thread_id]
    result = await Runner.run(agent, input, session=session)
    logger.success(format_json(result))
    return Turn(role="assistant", content=result.final_output)

"""
</TabItem>
<TabItem value="pydantic" label="Pydantic">
"""


agent = Agent('ollama:gpt-4', system_prompt="You are a helpful assistant that answers questions concisely.")

async def model_callback(input: str, turns: List[Turn]) -> Turn:
    message_history = []
    for turn in turns:
        if turn.role == "user":
            message_history.append(ModelRequest(parts=[UserPromptPart(content=turn.content, timestamp=datetime.now())], kind='request'))
        elif turn.role == "assistant":
            message_history.append(ModelResponse(parts=[TextPart(content=turn.content)], model_name='gpt-4', timestamp=datetime.now(), kind='response'))
    result = await agent.run(input, message_history=message_history)
    logger.success(format_json(result))
    return Turn(role="assistant", content=result.output)

"""
</TabItem>
</Tabs>

:::info
Your model callback should accept an `input`, and optionally `turns` and `thread_id`. It should return a `Turn` object.
:::

</TimelineItem>
<TimelineItem title="Simulate turns">

Use `deepeval`'s `ConversationSimulator` to simulate turns using goldens in your dataset:
"""
logger.info("Your model callback should accept an `input`, and optionally `turns` and `thread_id`. It should return a `Turn` object.")


simulator = ConversationSimulator(model_callback=chatbot_callback)
conversational_test_cases = simulator.simulate(goldens=dataset.goldens, max_turns=10)

"""
Here, we only have 1 test case, but in reality you'll want to simulate from at least 20 goldens.

<details>
<summary>Click to view an example simulated test case</summary>

Your generated test cases should be populated with simulated `Turn`s, along with the `scenario`, `expected_outcome`, and `user_description` from the conversation golden.
"""
logger.info("Here, we only have 1 test case, but in reality you'll want to simulate from at least 20 goldens.")

ConversationalTestCase(
    scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
    turns=[
        Turn(role="user", content="Hello, how are you?"),
        Turn(role="assistant", content="I'm doing well, thank you!"),
        Turn(role="user", content="How can I help you today?"),
        Turn(role="assistant", content="I'd like to buy a ticket to a Coldplay concert."),
    ]
)

"""
</details>

</TimelineItem>

<TimelineItem title="Run an evaluation">

Run an evaluation like how you learnt in the previous section:
"""
logger.info("Run an evaluation like how you learnt in the previous section:")

...

evaluate(conversational_test_cases, metrics=[TurnRelevancyMetric()])

"""
✅ Done. You've successfully learnt how to benchmark your chatbot.

<VideoDisplayer src="https://deepeval-docs.s3.us-east-1.amazonaws.com/getting-started%3Aconversation-test-report.mp4" />

</TimelineItem>

</Timeline>

## Next Steps

Now that you have run your first chatbot evals, you should:

1. **Customize your metrics**: Update the [list of metrics](/docs/metrics-introduction) based on your use case.
2. **Setup tracing**: It helps you [log multi-turn](https://www.confident-ai.com/docs/llm-tracing/advanced-features/threads) interactions in production.
3. **Enable evals in production**: Monitor performance over time [using the metrics](https://www.confident-ai.com/docs/llm-tracing/evaluations#offline-evaluations) you've defined on Confident AI.

You'll be able to analyze performance over time on **threads** this way, and add them back to your evals dataset for further evaluation.

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/llm-tracing:threads.mp4"
  confidentUrl="/docs/llm-tracing/evaluations#offline-evaluations"
  label="Chatbot Evals in Production"
/>
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)