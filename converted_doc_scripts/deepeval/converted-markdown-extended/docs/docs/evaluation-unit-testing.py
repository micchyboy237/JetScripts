from jet.transformers.formatters import format_json
from agents import Agent, Runner, SQLiteSession
from datetime import datetime
from deepeval import assert_test
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.dataset import EvaluationDataset
from deepeval.dataset import Golden
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import LLMTestCase
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
from main import chatbot_callback # Replace with your LLM callback
from ollama import Ollama
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from typing import List
from your_agent import your_llm_app # Replace with your LLM app
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import os
import pytest
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
id: evaluation-unit-testing-in-ci-cd
title: Unit Testing in CI/CD
sidebar_label: Unit Testing in CI/CD
---


Integrate LLM evaluations into your CI/CD pipeline with `deepeval` to catch regressions and ensure reliable performance. You can use `deepeval` with your CI/CD pipelines to run both end-to-end and component level evaluations.

`deepeval` allows you to run evaluations as if you're using `pytest` via our Pytest integration.

## End-to-End Evals in CI/CD

Run tests against your LLM app using golden datasets for every push you make. End-to-end evaluations validate overall behavior across single-turn and multi-turn interactions. Perfect for catching regressions before deploying to production.

### Single-Turn Evals

<Timeline>

<TimelineItem title="Load your dataset">

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("## End-to-End Evals in CI/CD")


dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

"""
</TabItem>
<TabItem value="csv" label="From CSV">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path="example.csv",
    input_col_name="query"
)

"""
</TabItem>
<TabItem value="json" label="From JSON">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path="example.json",
    input_key_name="query"
)

"""
</TabItem>
</Tabs>

You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).

</TimelineItem>
<TimelineItem title="Assert your tests">

You can use `deepeval`'s `assert_test` function to write test files.
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")




@pytest.mark.parametrize("golden",dataset.goldens)
def test_llm_app(golden: Golden):
    res, text_chunks = your_llm_app(golden.input)
    test_case = LLMTestCase(input=golden.input, actual_output=res, retrieval_context=text_chunks)
    assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])

@deepeval.log_hyperparameters(model="llama3.2", prompt_template="...")
def hyperparameters():
    return {"model": "gpt-4.1", "system prompt": "..."}

"""
Then, run the following command in your CLI:
"""
logger.info("Then, run the following command in your CLI:")

deepeval test run test_llm_app.py

"""
There are **TWO** mandatory and **ONE** optional parameter when calling the `assert_test()` function for **END-TO-END** evaluation:

- `test_case`: an `LLMTestCase`.
- `metrics`: a list of metrics of type `BaseMetric`.
- [Optional] `run_async`: a boolean which when set to `True`, enables concurrent evaluation of all metrics in `@observe`. Defaulted to `True`.

Create a YAML file to execute your test file automatically in CI/CD pipelines. [Click here for an example YAML file](#yaml-file-for-cicd-evals).

</TimelineItem>
</Timeline>


### Multi-Turn Evals

<Timeline>
<TimelineItem title="Wrap chatbot in callback">

You need to define a chatbot callback to generate synthetic test cases from goldens using the `ConversationSimulator`. So, define a callback function to generate the **next chatbot response** in a conversation, given the conversation history.

<Tabs groupId="techstack">
<TabItem value="python" label="Python">
"""
logger.info("### Multi-Turn Evals")


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
<TimelineItem title="Load your dataset">

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("Your model callback should accept an `input`, and optionally `turns` and `thread_id`. It should return a `Turn` object.")


dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

"""
</TabItem>
<TabItem value="csv" label="From CSV">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path="example.csv",
    input_col_name="query"
)

"""
</TabItem>
<TabItem value="json" label="From JSON">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path="example.json",
    input_key_name="query"
)

"""
</TabItem>
</Tabs>

You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).

</TimelineItem>
<TimelineItem title="Assert your tests">

You can use `deepeval`'s `assert_test` function to write test files.
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")




simulator = ConversationSimulator(model_callback=chatbot_callback)
conversational_test_cases = simulator.simulate(goldens=dataset.goldens, max_turns=10)

@pytest.mark.parametrize("test_case", conversational_test_cases)
def test_llm_app(test_case: ConversationalTestCase):
    assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])

@deepeval.log_hyperparameters(model="llama3.2", prompt_template="...")
def hyperparameters():
    return {"model": "gpt-4.1", "system prompt": "..."}

"""
Then, run the following command in your CLI:
"""
logger.info("Then, run the following command in your CLI:")

deepeval test run test_llm_app.py

"""
There are **TWO** mandatory and **ONE** optional parameter when calling the `assert_test()` function for **END-TO-END** evaluation:

- `test_case`: an `LLMTestCase`.
- `metrics`: a list of metrics of type `BaseMetric`.
- [Optional] `run_async`: a boolean which when set to `True`, enables concurrent evaluation of all metrics in `@observe`. Defaulted to `True`.

Create a YAML file to execute your test file automatically in CI/CD pipelines. [Click here for an example YAML file](#yaml-file-for-cicd-evals).

</TimelineItem>
</Timeline>

:::caution
The usual `pytest` command would still work but is highly not recommended. `deepeval test run` adds a range of functionalities on top of Pytest for unit-testing LLMs, which is enabled by [8+ optional flags](/docs/evaluation-flags-and-configs#flags-for-deepeval-test-run). Users typically include `deepeval test run` as a command in their `.yaml` files for pre-deployment checks in CI/CD pipelines ([example here](https://www.confident-ai.com/docs/llm-evaluation/unit-testing-cicd)).
:::

[Click here](/docs/evaluation-flags-and-configs#flags-for-deepeval-test-run) to learn about different optional flags available to `deepeval test run` to customize asynchronous behaviors, error handling, etc.


## Component-Level Evals in CI/CD

Test individual parts of your LLM pipeline like prompt templates or retrieval logic in isolation. Component-level evals offer fast, targeted feedback and integrate seamlessly into your CI/CD workflows.

<Timeline>
<TimelineItem title="Load your dataset">

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("## Component-Level Evals in CI/CD")


dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

"""
</TabItem>
<TabItem value="csv" label="From CSV">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path="example.csv",
    input_col_name="query"
)

"""
</TabItem>
<TabItem value="json" label="From JSON">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path="example.json",
    input_key_name="query"
)

"""
</TabItem>
</Tabs>

You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).

</TimelineItem>
<TimelineItem title="Assert your tests">

You can use `deepeval`'s `assert_test` function to write test files.
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")



@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden: Golden):
    assert_test(golden=golden, observed_callback=your_llm_app)

@deepeval.log_hyperparameters(model="llama3.2", prompt_template="...")
def hyperparameters():
    return {"model": "gpt-4.1", "system prompt": "..."}

"""
Finally, don't forget to run the test file in the CLI:
"""
logger.info("Finally, don't forget to run the test file in the CLI:")

deepeval test run test_llm_app.py

"""
There are **TWO** mandatory and **ONE** optional parameter when calling the `assert_test()` function for **COMPONENT-LEVEL** evaluation:

- `golden`: the `Golden` that you wish to invoke your `observed_callback` with.
- `observed_callback`: a function callback that is your `@observe` decorated LLM application. There must be **AT LEAST ONE** metric within one of the `metrics` in your `@observe` decorated LLM application.
- [Optional] `run_async`: a boolean which when set to `True`, enables concurrent evaluation of all metrics in `@observe`. Defaulted to `True`.

Create a YAML file to execute your test file automatically in CI/CD pipelines. [Click here for an example YAML file](#yaml-file-for-cicd-evals).

</TimelineItem>
</Timeline>

:::info
Similar to the `evaluate()` function, `assert_test()` for component-level evaluation does not need:

- Declaration of `metrics` because those are defined at the span level in the `metrics` parameter.
- Creation of `LLMTestCase`s because it is handled at runtime by `update_current_span` in your LLM app.
:::

## YAML File For CI/CD Evals

# To run your unit tests on all changes in prod, you can use the following `YAML` file in your **github actions** or any other similar CI/CD pipelines. This example uses `poetry` for installation, `OPENAI_API_KEY` as your LLM judge to run evals locally. You can also optionally add `CONFIDENT_API_KEY` to send results to Confident AI.
"""
logger.info("## YAML File For CI/CD Evals")

name: LLM App DeepEval Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install Dependencies
        run: poetry install --no-root

      - name: Run DeepEval Unit Tests
        env:
#           OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: poetry run deepeval test run test_llm_app.py

"""
[Click here](/docs/evaluation-flags-and-configs#flags-for-deepeval-test-run) to learn about different optional flags available to `deepeval test run` to customize asynchronous behaviors, error handling, etc.

:::tip
We highly recommend setting up [Confident AI](https://app.confident-ai.com) with your `deepeval` evaluations to get professional test reports and observe trends of your LLM application's performance overtime like this:

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/llm-tracing:spans.mp4"
  confidentUrl="/docs/llm-tracing/introduction"
  label="Span-Level Evals in Production"
/>
:::
"""
logger.info("We highly recommend setting up [Confident AI](https://app.confident-ai.com) with your `deepeval` evaluations to get professional test reports and observe trends of your LLM application's performance overtime like this:")

logger.info("\n\n[DONE]", bright=True)