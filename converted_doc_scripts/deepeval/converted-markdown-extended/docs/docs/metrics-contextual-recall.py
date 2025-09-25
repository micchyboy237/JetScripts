from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics.contextual_recall import ContextualRecallTemplate
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from jet.logger import logger
import Equation from "@site/src/components/Equation";
import MetricTagsDisplayer from "@site/src/components/MetricTagsDisplayer";
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
id: metrics-contextual-recall
title: Contextual Recall
sidebar_label: Contextual Recall
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-contextual-recall"
  />
</head>


<MetricTagsDisplayer singleTurn={true} rag={true} referenceBased={true} />

The contextual recall metric uses LLM-as-a-judge to measure the quality of your RAG pipeline's retriever by evaluating the extent of which the `retrieval_context` aligns with the `expected_output`. `deepeval`'s contextual recall metric is a self-explaining LLM-Eval, meaning it outputs a reason for its metric score.

:::info
Not sure if the `ContextualRecallMetric` is suitable for your use case? Run the follow command to find out:
"""
logger.info("id: metrics-contextual-recall")

deepeval recommend metrics

"""
:::

## Required Arguments

To use the `ContextualRecallMetric`, you'll have to provide the following arguments when creating an [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case):

- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`

Read the [How Is It Calculated](#how-is-it-calculated) section below to learn how test case parameters are used for metric calculation.

## Usage

The `ContextualRecallMetric()` can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluation:
"""
logger.info("## Required Arguments")


actual_output = "We offer a 30-day full refund at no extra cost."

expected_output = "You are eligible for a 30 day full refund at no extra cost."

retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = ContextualRecallMetric(
    threshold=0.7,
    model="llama3.2",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)


evaluate(test_cases=[test_case], metrics=[metric])

"""
There are **SEVEN** optional parameters when creating a `ContextualRecallMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.
- [Optional] `evaluation_template`: a class of type `ContextualRecallTemplate`, which allows you to [override the default prompts](#customize-your-template) used to compute the `ContextualRecallMetric` score. Defaulted to `deepeval`'s `ContextualRecallTemplate`.

### Within components

You can also run the `ContextualRecallMetric` within nested components for [component-level](/docs/evaluation-component-level-llm-evals) evaluation.
"""
logger.info("### Within components")

...

@observe(metrics=[metric])
def inner_component():
    test_case = LLMTestCase(input="...", actual_output="...")
    update_current_span(test_case=test_case)
    return

@observe
def llm_app(input: str):
    inner_component()
    return

evaluate(observed_callback=llm_app, goldens=[Golden(input="Hi!")])

"""
### As a standalone

You can also run the `ContextualRecallMetric` on a single test case as a standalone, one-off execution.
"""
logger.info("### As a standalone")

...

metric.measure(test_case)
logger.debug(metric.score, metric.reason)

"""
:::caution
This is great for debugging or if you wish to build your own evaluation pipeline, but you will **NOT** get the benefits (testing reports, Confident AI platform) and all the optimizations (speed, caching, computation) the `evaluate()` function or `deepeval test run` offers.
:::

## How Is It Calculated?

The `ContextualRecallMetric` score is calculated according to the following equation:

<Equation formula="\text{Contextual Recall} = \frac{\text{Number of Attributable Statements}}{\text{Total Number of Statements}}" />

The `ContextualRecallMetric` first uses an LLM to extract all **statements made in the `expected_output`**, before using the same LLM to classify whether each statement can be attributed to nodes in the `retrieval_context`.

:::info
We use the `expected_output` instead of the `actual_output` because we're measuring the quality of the RAG retriever for a given ideal output.
:::

A higher contextual recall score represents a greater ability of the retrieval system to capture all relevant information from the total available relevant set within your knowledge base.

## Customize Your Template

Since `deepeval`'s `ContextualRecallMetric` is evaluated by LLM-as-a-judge, you can likely improve your metric accuracy by [overriding `deepeval`'s default prompt templates](/docs/metrics-introduction#customizing-metric-prompts). This is especially helpful if:

- You're using a [custom evaluation LLM](/guides/guides-using-custom-llms), especially for smaller models that have weaker instruction following capabilities.
- You want to customize the examples used in the default `ContextualRecallTemplate` to better align with your expectations.

:::tip
You can learn what the default `ContextualRecallTemplate` looks like [here on GitHub](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/contextual_recall/template.py), and should read the [How Is It Calculated](#how-is-it-calculated) section above to understand how you can tailor it to your needs.
:::

Here's a quick example of how you can override the relevancy classification step of the `ContextualRecallMetric` algorithm:
"""
logger.info("## How Is It Calculated?")


class CustomTemplate(ContextualRecallTemplate):
    @staticmethod
    def generate_verdicts(expected_output: str, retrieval_context: List[str]):
        return f"""For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the nodes of retrieval contexts.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
    ]
}}

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""

metric = ContextualRecallMetric(evaluation_template=CustomTemplate)
metric.measure(...)

logger.info("\n\n[DONE]", bright=True)