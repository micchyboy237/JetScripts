from jet.transformers.formatters import format_json
from agents import Agent, Runner, SQLiteSession
from datetime import datetime
from deepeval import evaluate
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.dataset import EvaluationDataset
from deepeval.dataset import Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import TurnRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case import Turn
from deepeval.tracing import observe, update_current_trace
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
from your_agent import your_llm_app # Replace with your LLM app
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
id: evaluation-end-to-end-llm-evals
title: End-to-End LLM Evaluation
sidebar_label: End-to-End Evals
---


End-to-end evaluation assesses the "observable" inputs and outputs of your LLM application - it is what users see, and treats your LLM application as a black-box.

![end-to-end evals](https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:end-to-end-llm-evals.png)

<details>
<summary><strong>When should you run End-to-End evaluations?</strong></summary>

For simple LLM applications like basic RAG pipelines with "flat" architectures
that can be represented by a single <code>LLMTestCase</code>, end-to-end
evaluation is ideal. Common use cases that are suitable for end-to-end
evaluation include (not inclusive):

- RAG QA
- PDF extraction
- Writing assitants
- Summarization
- etc.

You'll notice that use cases with simplier architectures are more suited for end-to-end evaluation. However, if your system is an extremely complex agentic workflow, you might also find end-to-end evaluation more suitable as you'll might conclude that that component-level evaluation gives you too much noise in its evaluation results.

Most of what you saw in DeepEval's <a href="/docs/getting-started">quickstart</a> is end-to-end evaluation!

</details>

## What Are E2E Evals

Running an end-to-end LLM evaluation creates a **test run** — a collection of test cases that benchmarks your LLM application at a specific point in time. You would typically:

- Loop through a list of `Golden`s
- Invoke your LLM app with each golden's `input`
- Generate a set of test cases ready for evaluation
- Apply metrics to your test cases and run evaluations

:::info
To get a more fully sharable [LLM test report](https://www.confident-ai.com/docs/llm-evaluation/dashboards/testing-reports) login to Confident AI [here](https://app.confident-ai.com) or run the following in your terminal:
"""
logger.info("## What Are E2E Evals")

deepeval login

"""
:::

## Setup Your Test Environment

<Timeline>
<TimelineItem title="Create a dataset">

[Datasets](/docs/evaluation-datasets) in `deepeval` allow you to store [`Golden`](/docs/evaluation-datasets#what-are-goldens)s, which are like a precursors to test cases. They allow you to create test case dynamically during evaluation time by calling your LLM application. Here's how you can create goldens:

<Tabs>
<TabItem label="Single-Turn" value="single-turn">
"""
logger.info("## Setup Your Test Environment")


goldens=[
    Golden(input="What is your name?"),
    Golden(input="Choose a number between 1 to 100"),
    ...
]

"""
</TabItem>
<TabItem label="Multi-Turn" value="multi-turn">
"""


goldens = [
    ConversationalGolden(
        scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
        expected_outcome="Successful purchase of a ticket.",
        user_description="Andy Byron is the CEO of Astronomer.",
    ),
    ...
]

"""
</TabItem>
</Tabs>

You can also generate synthetic goldens automatically using the `Synthesizer`. Learn more [here](/docs/synthesizer-introduction). You can now use these goldens to create an evaluation dataset that can be stored and loaded them anytime.

Here's an example showing how you can create and store datasets in `deepeval`:

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("You can also generate synthetic goldens automatically using the `Synthesizer`. Learn more [here](/docs/synthesizer-introduction). You can now use these goldens to create an evaluation dataset that can be stored and loaded them anytime.")


dataset = EvaluationDataset(goldens)
dataset.push(alias="My dataset")

"""
</TabItem>
<TabItem value="csv" label="Locally as CSV">
"""


dataset = EvaluationDataset(goldens)
dataset.save_as(
    file_type="csv",
    directory="./example"
)

"""
</TabItem>
<TabItem value="json" label="Locally as JSON">
"""


dataset = EvaluationDataset(goldens)
dataset.save_as(
    file_type="json",
    directory="./example"
)

"""
</TabItem>
</Tabs>

✅ Done. You can now use this dataset anywhere to run your evaluations automatically by looping over them and generating test cases.

</TimelineItem>
<TimelineItem title="Select metrics">

When it comes to selecting metrics for your application, we recommend choosing no more than 5 metrics, comprising of:

- (2 - 3) **Generic metrics** for your application type. (_e.g. Agents, RAG, Chabot_)
- (1 - 2) **Custom metrics** for your specific use case.

You can read our [metrics section](/docs/metrics-introduction) to learn about the 40+ metrics we offer. Or come to [our discord](https://discord.com/invite/a3K9c8GRGt) and get some tailored recommendations from our team.

</TimelineItem>
</Timeline>

You can now use these test cases and metrics to run [single-turn](#single-turn-end-to-end-evals) and [multi-turn](#multi-turn-end-to-end-evals) end-to-end evals. If you've setup [tracing](/docs/evaluation-llm-tracing) for your LLM application, you can automatically [run end-to-end evals for traces](#end-to-end-evals-for-tracing) using a single line of code.

## Single-Turn E2E Evals

<Timeline>
<TimelineItem title="Load your dataset">

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("## Single-Turn E2E Evals")


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
<TimelineItem title="Create test cases using dataset">

You can now create `LLMTestCase`s using the goldens by calling your LLM application.
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")


dataset = EvaluationDataset()

test_cases = []

for golden in dataset.goldens:
    res, text_chunks = your_llm_app(golden.input)
    test_case = LLMTestCase(input=golden.input, actual_output=res, retrieval_context=text_chunks)
    test_cases.append(test_case)

"""
You can also add test cases directly into your dataset by using the `add_test_case()` method.

</TimelineItem>
<TimelineItem title="Run end-to-end evals">

You should pass the `test_cases` and `metrics` you've decided in the `evaluate()` function to run end-to-end evals.
"""
logger.info("You can also add test cases directly into your dataset by using the `add_test_case()` method.")

...

evaluate(
    test_cases=test_cases,
    metrics=[AnswerRelevancyMetric()],
    hyperparameters={
        model="llama3.2",
        system_prompt="..."
    }
)

"""
There are **TWO** mandatory and **SIX** optional parameters when calling the `evaluate()` function for **END-TO-END** evaluation:

- `test_cases`: a list of `LLMTestCase`s **OR** `ConversationalTestCase`s, or an `EvaluationDataset`. You cannot evaluate `LLMTestCase`/`MLLMTestCase`s and `ConversationalTestCase`s in the same test run.
- `metrics`: a list of metrics of type `BaseMetric`.
- [Optional] `hyperparameters`: a dict of type `dict[str, Union[str, int, float]]`. You can log any arbitrary hyperparameter associated with this test run to pick the best hyperparameters for your LLM application on Confident AI.
- [Optional] `identifier`: a string that allows you to better identify your test run on Confident AI.
- [Optional] `async_config`: an instance of type `AsyncConfig` that allows you to [customize the degree of concurrency](/docs/evaluation-flags-and-configs#async-configs) during evaluation. Defaulted to the default `AsyncConfig` values.
- [Optional] `display_config`:an instance of type `DisplayConfig` that allows you to [customize what is displayed](/docs/evaluation-flags-and-configs#display-configs) to the console during evaluation. Defaulted to the default `DisplayConfig` values.
- [Optional] `error_config`: an instance of type `ErrorConfig` that allows you to [customize how to handle errors](/docs/evaluation-flags-and-configs#error-configs) during evaluation. Defaulted to the default `ErrorConfig` values.
- [Optional] `cache_config`: an instance of type `CacheConfig` that allows you to [customize the caching behavior](/docs/evaluation-flags-and-configs#cache-configs) during evaluation. Defaulted to the default `CacheConfig` values.

This is exactly the same as `assert_test()` in `deepeval test run`, but in a different interface.

</TimelineItem>
</Timeline>

:::tip
We recommend logging your `hyperparameters` during your evauations as they allow you find the best model configuration for your application.

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/evaluation:parameter-insights.mp4"
  confidentUrl="https://www.confident-ai.com/docs/llm-evaluation/dashboards/model-and-prompt-insights"
  label="Parameter Insights To Find Best Model"
/>
:::

## Multi-Turn E2E Evals

<Timeline>
<TimelineItem title="Wrap chatbot in callback">

You need to define a chatbot callback to generate synthetic test cases from goldens using the `ConversationSimulator`. So, define a callback function to generate the **next chatbot response** in a conversation, given the conversation history.

<Tabs groupId="techstack">
<TabItem value="python" label="Python">
"""
logger.info("## Multi-Turn E2E Evals")


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

<TabItem value="from-json" label="From JSON">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path="example.json",
    input_key_name="query"
)

"""
</TabItem>

<TabItem value="from-csv" label="From CSV">
"""


dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path="example.csv",
    input_col_name="query"
)

"""
</TabItem>
</Tabs>

You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).

</TimelineItem>
<TimelineItem title="Simulate turns">

Use `deepeval`'s `ConversationSimulator` to simulate turns using goldens in your dataset:
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")


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

evaluate(
  conversational_test_cases,
  metrics=[TurnRelevancyMetric()],
  hyperparameters={
      model="llama3.2",
      system_prompt="..."
  }
)

"""
There are **TWO** mandatory and **SIX** optional parameters when calling the `evaluate()` function for **END-TO-END** evaluation:

- `test_cases`: a list of `LLMTestCase`s **OR** `ConversationalTestCase`s, or an `EvaluationDataset`. You cannot evaluate `LLMTestCase`/`MLLMTestCase`s and `ConversationalTestCase`s in the same test run.
- `metrics`: a list of metrics of type `BaseConversationalMetric`.
- [Optional] `hyperparameters`: a dict of type `dict[str, Union[str, int, float]]`. You can log any arbitrary hyperparameter associated with this test run to pick the best hyperparameters for your LLM application on Confident AI.
- [Optional] `identifier`: a string that allows you to better identify your test run on Confident AI.
- [Optional] `async_config`: an instance of type `AsyncConfig` that allows you to [customize the degree of concurrency](/docs/evaluation-flags-and-configs#async-configs) during evaluation. Defaulted to the default `AsyncConfig` values.
- [Optional] `display_config`:an instance of type `DisplayConfig` that allows you to [customize what is displayed](/docs/evaluation-flags-and-configs#display-configs) to the console during evaluation. Defaulted to the default `DisplayConfig` values.
- [Optional] `error_config`: an instance of type `ErrorConfig` that allows you to [customize how to handle errors](/docs/evaluation-flags-and-configs#error-configs) during evaluation. Defaulted to the default `ErrorConfig` values.
- [Optional] `cache_config`: an instance of type `CacheConfig` that allows you to [customize the caching behavior](/docs/evaluation-flags-and-configs#cache-configs) during evaluation. Defaulted to the default `CacheConfig` values.

This is exactly the same as `assert_test()` in `deepeval test run`, but in a difference interface.

</TimelineItem>
</Timeline>

We highly recommend setting up [Confident AI](https://app.confident-ai.com) with your `deepeval` evaluations to get professional test reports and observe trends of your LLM application's performance overtime like this:

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/evaluation:multi-turn-e2e-report.mp4"
  confidentUrl="https://www.confident-ai.com/docs/llm-evaluation/dashboards/testing-reports"
  label="Test Reports After Running Evals on Confident AI"
/>

## E2E Evals For Tracing

If you've [setup tracing](/docs/evaluation-llm-tracing) for you LLM application, you can run end-to-end evals using the `evals_iterator()` function.

<Timeline>
<TimelineItem title="Load your dataset">

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>
<TabItem value="confident-ai" label="Confident AI">
"""
logger.info("## E2E Evals For Tracing")


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

<TimelineItem title="Update your test cases for trace">

You can update your end-to-end test cases for trace by using the `update_current_trace` function provided by `deepeval`
"""
logger.info("You can [learn more about loading datasets here](/docs/evaluation-datasets#load-dataset).")


@observe()
def llm_app(query: str) -> str:

    @observe()
    def retriever(query: str) -> list[str]:
        chunks = ["List", "of", "text", "chunks"]
        update_current_trace(retrieval_context=chunks)
        return chunks

    @observe()
    def generator(query: str, text_chunks: list[str]) -> str:
        res = Ollama().chat.completions.create(model="llama3.2", messages=[{"role": "user", "content": query}]
        ).choices[0].message.content
        update_current_trace(input=query, output=res)
        return res

    return generator(query, retriever(query))

"""
There are **TWO** ways to create test cases when using the `update_current_trace` function:

- [Optional] `test_case`: Takes an `LLMTestCase` to create a span level test case for that component.

- Or, You can also opt to give the values of `LLMTestCase` directly by using the following attributes:
  - [Optional] `input`
  - [Optional] `output`
  - [Optional] `retrieval_context`
  - [Optional] `context`
  - [Optional] `expected_output`
  - [Optional] `tools_called`
  - [Optional] `expected_tools`

:::note
You can use the individual `LLMTestCase` params in the `update_current_trace` function to override the values of the `test_case` you passed.
:::

</TimelineItem>
<TimelineItem title="Run end-to-end evals">

You can run end-to-end evals for your traces by supplying your `metrics` in the `evals_iterator` function.
"""
logger.info("There are **TWO** ways to create test cases when using the `update_current_trace` function:")


dataset = EvaluationDataset()
dataset.pull(alias="YOUR-DATASET-ALIAS")

for golden in dataset.evals_iterator(metrics=[AnswerRelevancyMetric()]):
    llm_app(golden.input) # Replace with your LLM app

"""
There are **SIX** optional parameters when using the `evals_iterator()`:

- [Optional] `metrics`: a list of `BaseMetric` that allows you to run end-to-end evals for your traces.
- [Optional] `identifier`: a string that allows you to better identify your test run on Confident AI.
- [Optional] `async_config`: an instance of type `AsyncConfig` that allows you to [customize the degree concurrency](/docs/evaluation-flags-and-configs#async-configs) during evaluation. Defaulted to the default `AsyncConfig` values.
- [Optional] `display_config`:an instance of type `DisplayConfig` that allows you to [customize what is displayed](/docs/evaluation-flags-and-configs#display-configs) to the console during evaluation. Defaulted to the default `DisplayConfig` values.
- [Optional] `error_config`: an instance of type `ErrorConfig` that allows you to [customize how to handle errors](/docs/evaluation-flags-and-configs#error-configs) during evaluation. Defaulted to the default `ErrorConfig` values.
- [Optional] `cache_config`: an instance of type `CacheConfig` that allows you to [customize the caching behavior](/docs/evaluation-flags-and-configs#cache-configs) during evaluation. Defaulted to the default `CacheConfig` values.

This is all it takes to run end-to-end evaluations, with the added benefit of a full testing report with tracing included on Confident AI.

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/evaluation:single-turn-e2e-report-tracing.mp4"
  confidentUrl="https://www.confident-ai.com/docs/llm-evaluation/dashboards/testing-reports"
  label="Test Reports For Evals and Traces on Confident AI"
/>

</TimelineItem>
</Timeline>

If you want to run end-to-end evaluations in CI/CD piplines, [click here](/docs/evaluation-unit-testing-in-ci-cd#end-to-end-evals-in-cicd).
"""
logger.info("There are **SIX** optional parameters when using the `evals_iterator()`:")

logger.info("\n\n[DONE]", bright=True)