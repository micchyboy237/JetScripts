from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import PromptAlignmentMetric
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
id: metrics-prompt-alignment
title: Prompt Alignment
sidebar_label: Prompt Alignment
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-prompt-alignment"
  />
</head>


<MetricTagsDisplayer singleTurn={true} referenceless={true} />

The prompt alignment metric uses LLM-as-a-judge to measure whether your LLM application is able to generate `actual_output`s that aligns with any **instructions** specified in your prompt template. `deepeval`'s prompt alignment metric is a self-explaining LLM-Eval, meaning it outputs a reason for its metric score.

:::tip
Not sure if this metric is for you? Run the follow command to find out:
"""
logger.info("id: metrics-prompt-alignment")

deepeval recommend metrics

"""
:::

## Required Arguments

To use the `PromptAlignmentMetric`, you'll have to provide the following arguments when creating an [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case):

- `input`
- `actual_output`

Read the [How Is It Calculated](#how-is-it-calculated) section below to learn how test case parameters are used for metric calculation.

## Usage

The `PromptAlignmentMetric()` can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluation:
"""
logger.info("## Required Arguments")


metric = PromptAlignmentMetric(
    prompt_instructions=["Reply in all uppercase"],
    model="llama3.2",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost."
)


evaluate(test_cases=[test_case], metrics=[metric])

"""
There are **ONE** mandatory and **SIX** optional parameters when creating an `PromptAlignmentMetric`:

- `prompt_instructions`: a list of strings specifying the instructions you want followed in your prompt template.
- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-a-metric-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

### Within components

You can also run the `PromptAlignmentMetric` within nested components for [component-level](/docs/evaluation-component-level-llm-evals) evaluation.
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

You can also run the `PromptAlignmentMetric` on a single test case as a standalone, one-off execution.
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

The `PromptAlignmentMetric` score is calculated according to the following equation:

<Equation formula="\text{Prompt Alignment} = \frac{\text{Number of Instructions Followed}}{\text{Total Number of Instructions}}" />

The `PromptAlignmentMetric` uses an LLM to classify whether each prompt instruction is followed in the `actual_output` using additional context from the `input`.

:::tip

By providing an initial list of `prompt_instructions` instead of the entire prompt template, the `PromptAlignmentMetric` is able to more accurately determine whether the core instructions laid out in your prompt template is followed.

:::
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)