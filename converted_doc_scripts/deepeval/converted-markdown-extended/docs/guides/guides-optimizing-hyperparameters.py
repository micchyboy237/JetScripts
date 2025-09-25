from deepeval import assert_test
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from jet.logger import logger
from typing import List
import deepeval
import os
import pytest
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
# id: guides-optimizing-hyperparameters
title: Optimizing Hyperparameters for LLM Applications
sidebar_label: Optimizing Hyperparameters
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/guides/guides-optimizing-hyperparameters"
  />
</head>

Apart from catching regressions and sanity checking your LLM applications, LLM evaluation and testing plays an pivotal role in picking the best hyperparameters for your LLM application.

:::info
In `deepeval`, hyperparameters refer to independent variables that affect the final `actual_output` of your LLM application, which includes the LLM used, the prompt template, temperature, etc.
:::

## Which Hyperparameters Should I Iterate On?

Here are typically the hyperparameters you should iterate on:

- **model**: the LLM to use for generation.
- **prompt template**: the variation of prompt templates to use for generation.
- **temperature**: the temperature value to use for generation.
- **max tokens**: the max token limit to set for your LLM generation.
- **top-K**: the number of retrieved nodes in your `retrieval_context` in a RAG pipeline.
- **chunk size**: the size of the retrieved nodes in your `retrieval_context` in a RAG pipeline.
- **reranking model**: the model used to rerank the retrieved nodes in your `retrieval_context` in a RAG pipeline.

:::tip
In the previous guide on [RAG Evaluation](/guides/guides-rag-evaluation), you already saw how `deepeval`'s RAG metrics can help iterate on many of the hyperparameters used within a RAG pipeline.
:::

## Finding The Best Hyperparameter Combination

To find the best hyperparameter combination, simply:

- choose a/multiple [LLM evaluation metrics](#metrics-introduction) that fits your evaluation criteria
- execute evaluations in a nested for-loop, while generating `actual_outputs` **at evaluation time** based on the current hyperparameter combination

:::note
In reality, you don't have to strictly generate `actual_outputs` at evaluation time and can evaluate with datasets of precomputed `actual_outputs`, but you ought to ensure that the `actual_outputs` in each [`LLMTestCase`](/docs/evaluation-test-cases) can be properly identified by a hyperparameter combination for this to work.
:::

Let's walkthrough a quick example hypothetical example showing how to find the best model and prompt template hyperparameter combination using the `AnswerRelevancyMetric` as a measurement. First, define a function to generate `actual_output`s for `LLMTestCase`s based on a certain hyperparameter combination:
"""
logger.info("# id: guides-optimizing-hyperparameters")


def construct_test_cases(model: str, prompt_template: str) : List[LLMTestCase]:
    prompt = format_prompt_template(prompt_template)
    llm = get_llm(model)

    test_cases : List[LLMTestCase] = []
    for input in list_of_inputs:
        test_case = LLMTestCase(
            input=input,
            actual_output=generate_actual_output(llm, prompt)
        )
        test_cases.append(test_case)

    return test_cases

"""
:::info
You **should definitely try** logging into Confident AI before continuing to the final step. Confident AI allows you to search, filter for, and view metric evaluation results on the web to pick the best hyperparameter combination for your LLM application.

Simply run `deepeval login`:
"""
logger.info("You **should definitely try** logging into Confident AI before continuing to the final step. Confident AI allows you to search, filter for, and view metric evaluation results on the web to pick the best hyperparameter combination for your LLM application.")

deepeval login

"""
:::

Then, define the `AnswerRelevancyMetric` and use this helper function to construct `LLMTestCase`s:
"""
logger.info("Then, define the `AnswerRelevancyMetric` and use this helper function to construct `LLMTestCase`s:")

...

metric = AnswerRelevancyMetric()

for model in models:
    for prompt_template in prompt_templates:
        evaluate(
            test_cases=construct_test_cases(model, prompt_template),
            metrics=[metric],
            hyperparameter={
                "model": model,
                "prompt template": prompt_template
            }
        )

"""
:::tip
Remember, we're just using the `AnswerRelevancyMetric` as an example here and you should choose whichever [LLM evaluation metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation) based on whatever custom criteria you want to assess your LLM application on.
:::

## Keeping Track of Hyperparameters in CI/CD

You can also keep track of hyperparameters used during testing in your CI/CD pipelines. This is helpful since you will be able to pinpoint the hyperparameter combination associated with failing test runs.

To begin, login to Confident AI:
"""
logger.info("## Keeping Track of Hyperparameters in CI/CD")

deepeval login

"""
Then define your test function and log hyperparameters in your test file:
"""
logger.info("Then define your test function and log hyperparameters in your test file:")


test_cases = [...]

@pytest.mark.parametrize(
    "test_case",
    test_cases,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])


@deepeval.log_hyperparameters(model="llama3.2", prompt_template="...")
def hyperparameters():
    return {
        "temperature": 1,
        "chunk size": 500
    }

"""
Lastly, run `deepeval test run`:
"""
logger.info("Lastly, run `deepeval test run`:")

deepeval test run test_file.py

"""
In the next guide, we'll show you to build your own custom LLM evaluation metrics in case you want more control over evaluation when picking for hyperparameters.
"""
logger.info("In the next guide, we'll show you to build your own custom LLM evaluation metrics in case you want more control over evaluation when picking for hyperparameters.")

logger.info("\n\n[DONE]", bright=True)