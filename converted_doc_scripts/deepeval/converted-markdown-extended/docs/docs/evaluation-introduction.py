from deepeval import assert_test
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from jet.logger import logger
from ollama import Ollama
import os
import shutil


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
id: evaluation-introduction
title: LLM Evals Introduction
sidebar_label: Introduction
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/evaluation-introduction"
  />
</head>

## Quick Summary

Evaluation refers to the process of testing your LLM application outputs, and requires the following components:

- Test cases
- Metrics
- Evaluation dataset

Here's a diagram of what an ideal evaluation workflow looks like using `deepeval`:

<img
  id="invertable-img"
  src="https://d2lsxfc3p6r9rv.cloudfront.net/workflow.png"
  style={{ padding: "30px", marginTop: "20px" }}
/>

There are **TWO** types of LLM evaluations in `deepeval`:

- [End-to-end evaluation](/docs/evaluation-end-to-end-llm-evals): The overall input and outputs of your LLM system.

- [Component-level evaluation](/docs/evaluation-component-level-llm-evals): The individual inner workings of your LLM system.

Both can be done using either `deepeval test run` in CI/CD pipelines, or via the `evaluate()` function in Python scripts.

:::note
Your test cases will typically be in a single python file, and executing them will be as easy as running `deepeval test run`:

deepeval test run test_example.py

:::

## Test Run

Running an LLM evaluation creates a **test run** — a collection of test cases that benchmarks your LLM application at a specific point in time. If you're logged into Confident AI, you'll also receive a fully sharable [LLM testing report](https://www.confident-ai.com/docs/llm-evaluation/dashboards/testing-reports) on the cloud.

## Metrics

`deepeval` offers 30+ evaluation metrics, most of which are evaluated using LLMs (visit the [metrics section](/docs/metrics-introduction#types-of-metrics) to learn why).


answer_relevancy_metric = AnswerRelevancyMetric()

You'll need to create a test case to run `deepeval`'s metrics.

## Test Cases

In `deepeval`, a test case represents an [LLM interaction](/docs/evaluation-test-cases#what-is-an-llm-interaction) and allows you to use evaluation metrics you have defined to unit test LLM applications.
"""
logger.info("## Quick Summary")


test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

"""
In this example, `input` mimics an user interaction with a RAG-based LLM application, where `actual_output` is the output of your LLM application and `retrieval_context` is the retrieved nodes in your RAG pipeline. Creating a test case allows you to evaluate using `deepeval`'s default metrics:
"""
logger.info("In this example, `input` mimics an user interaction with a RAG-based LLM application, where `actual_output` is the output of your LLM application and `retrieval_context` is the retrieved nodes in your RAG pipeline. Creating a test case allows you to evaluate using `deepeval`'s default metrics:")


answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

answer_relevancy_metric.measure(test_case)
logger.debug(answer_relevancy_metric.score)

"""
## Datasets

Datasets in `deepeval` is a collection of goldens. It provides a centralized interface for you to evaluate a collection of test cases using one or multiple metrics.
"""
logger.info("## Datasets")


answer_relevancy_metric = AnswerRelevancyMetric()
dataset = EvaluationDataset(goldens=[Golden(input="Who is the current president of the United States of America?")])

for golden in dataset.goldens:
  dataset.add_test_case(
    LLMTestCase(
      input=golden.input,
      actual_output=you_llm_app(golden.input)
    )
  )

evaluate(test_cases=dataset.test_cases, metrics=[answer_relevancy_metric])

"""
:::note
You don't need to create an evaluation dataset to evaluate individual test cases. Visit the [test cases section](/docs/evaluation-test-cases#assert-a-test-case) to learn how to assert individual test cases.
:::

## Synthesizer

In `deepeval`, the `Synthesizer` allows you to generate synthetic datasets. This is especially helpful if you don't have production data or you don't have a golden dataset to evaluate with.
"""
logger.info("## Synthesizer")


synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
  document_paths=['example.txt', 'example.docx', 'example.pdf']
)

dataset = EvaluationDataset(goldens=goldens)

"""
:::info
`deepeval`'s `Synthesizer` is highly customizable, and you can learn more about it [here.](/docs/synthesizer-introduction)
:::

## Evaluating With Pytest

:::caution
Although `deepeval` integrates with Pytest, we highly recommend you to **AVOID** executing `LLMTestCase`s directly via the `pytest` command to avoid any unexpected errors.
:::

`deepeval` allows you to run evaluations as if you're using Pytest via our Pytest integration. Simply create a test file:
"""
logger.info("## Evaluating With Pytest")


dataset = EvaluationDataset(goldens=[...])

for golden in dataset.goldens:
  dataset.add_test_case(...) # convert golden to test case

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    assert_test(test_case, [AnswerRelevancyMetric()])

"""
And run the test file in the CLI using `deepeval test run`:
"""
logger.info("And run the test file in the CLI using `deepeval test run`:")

deepeval test run test_example.py

"""
There are **TWO** mandatory and **ONE** optional parameter when calling the `assert_test()` function:

- `test_case`: an `LLMTestCase`
- `metrics`: a list of metrics of type `BaseMetric`
- [Optional] `run_async`: a boolean which when set to `True`, enables concurrent evaluation of all metrics. Defaulted to `True`.

You can find the full documentation on `deepeval test run`, for both [end-to-end](/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines) and [component-level](/docs/evaluation-component-level-llm-evals#use-deepeval-test-run-in-cicd-pipelines) evaluation by clicking on their respective links.

:::info
`@pytest.mark.parametrize` is a decorator offered by Pytest. It simply loops through your `EvaluationDataset` to evaluate each test case individually.
:::

You can include the `deepeval test run` command as a step in a `.yaml` file in your CI/CD workflows to run pre-deployment checks on your LLM application.

## Evaluating Without Pytest

Alternately, you can use `deepeval`'s `evaluate` function. This approach avoids the CLI (if you're in a notebook environment), and allows for parallel test execution as well.
"""
logger.info("## Evaluating Without Pytest")


dataset = EvaluationDataset(goldens=[...])
for golden in dataset.goldens:
  dataset.add_test_case(...) # convert golden to test case

evaluate(dataset, [AnswerRelevancyMetric()])

"""
There are **TWO** mandatory and **SIX** optional parameters when calling the `evaluate()` function:

- `test_cases`: a list of `LLMTestCase`s **OR** `ConversationalTestCase`s, or an `EvaluationDataset`. You cannot evaluate `LLMTestCase`/`MLLMTestCase`s and `ConversationalTestCase`s in the same test run.
- `metrics`: a list of metrics of type `BaseMetric`.
- [Optional] `hyperparameters`: a dict of type `dict[str, Union[str, int, float]]`. You can log any arbitrary hyperparameter associated with this test run to pick the best hyperparameters for your LLM application on Confident AI.
- [Optional] `identifier`: a string that allows you to better identify your test run on Confident AI.
- [Optional] `async_config`: an instance of type `AsyncConfig` that allows you to [customize the degree concurrency](/docs/evaluation-flags-and-configs#async-configs) during evaluation. Defaulted to the default `AsyncConfig` values.
- [Optional] `display_config`:an instance of type `DisplayConfig` that allows you to [customize what is displayed](/docs/evaluation-flags-and-configs#display-configs) to the console during evaluation. Defaulted to the default `DisplayConfig` values.
- [Optional] `error_config`: an instance of type `ErrorConfig` that allows you to [customize how to handle errors](/docs/evaluation-flags-and-configs#error-configs) during evaluation. Defaulted to the default `ErrorConfig` values.
- [Optional] `cache_config`: an instance of type `CacheConfig` that allows you to [customize the caching behavior](/docs/evaluation-flags-and-configs#cache-configs) during evaluation. Defaulted to the default `CacheConfig` values.

You can find the full documentation on `evaluate()`, for both [end-to-end](/docs/evaluation-end-to-end-llm-evals#use-evaluate-in-python-scripts) and [component-level](/docs/evaluation-component-level-llm-evals#use-evaluate-in-python-scripts) evaluation by clicking on their respective links.

:::tip
You can also replace `dataset` with a list of test cases, as shown in the [test cases section.](/docs/evaluation-test-cases#evaluate-test-cases-in-bulk)
:::

## Evaluating Nested Components

You can also run metrics on nested components by setting up tracing in `deepeval`, and requires under 10 lines of code:
"""
logger.info("## Evaluating Nested Components")


client = Ollama()

@observe(metrics=[AnswerRelevancyMetric()])
def complete(query: str):
  response = client.chat.completions.create(model="llama3.2", messages=[{"role": "user", "content": query}]).choices[0].message.content

  update_current_span(
    test_case=LLMTestCase(input=query, output=response)
  )
  return response

"""
This is very useful especially if you:

- Want to run a different set of metrics on different components
- Wish to evaluate multiple components at once
- Don't want to rewrite your codebase just to bubble up returned variables to create an `LLMTestCase`

By defauly, `deepeval` will not run any metrics when you're running your LLM application outside of `evaluate()` or `assert_test()`. For the full guide on evaluating with tracing, visit [this page.](/docs/evaluation-component-level-llm-evals)
"""
logger.info("This is very useful especially if you:")

logger.info("\n\n[DONE]", bright=True)