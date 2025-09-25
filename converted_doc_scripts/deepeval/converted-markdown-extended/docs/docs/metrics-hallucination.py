from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import HallucinationMetric
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
id: metrics-hallucination
title: Hallucination
sidebar_label: Hallucination
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-hallucination"
  />
</head>


<MetricTagsDisplayer singleTurn={true} referenceBased={true} />

The hallucination metric uses LLM-as-a-judge to determine whether your LLM generates factually correct information by comparing the `actual_output` to the provided `context`.

:::info
If you're looking to evaluate hallucination for a RAG system, please refer to the [faithfulness metric](/docs/metrics-faithfulness) instead.
:::

## Required Arguments

To use the `HallucinationMetric`, you'll have to provide the following arguments when creating an [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case):

- `input`
- `actual_output`
- `context`

Read the [How Is It Calculated](#how-is-it-calculated) section below to learn how test case parameters are used for metric calculation.

## Usage

The `HallucinationMetric()` can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluation:
"""
logger.info("## Required Arguments")


context=["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

actual_output="A blond drinking water in public."

test_case = LLMTestCase(
    input="What was the blond doing?",
    actual_output=actual_output,
    context=context
)
metric = HallucinationMetric(threshold=0.5)


evaluate(test_cases=[test_case], metrics=[metric])

"""
There are **SIX** optional parameters when creating a `HallucinationMetric`:

- [Optional] `threshold`: a float representing the maximum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 0 for perfection, 1 otherwise. It also overrides the current threshold and sets it to 0. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

### Within components

You can also run the `HallucinationMetric` within nested components for [component-level](/docs/evaluation-component-level-llm-evals) evaluation.
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

You can also run the `HallucinationMetric` on a single test case as a standalone, one-off execution.
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

The `HallucinationMetric` score is calculated according to the following equation:

<Equation formula="\text{Hallucination} = \frac{\text{Number of Contradicted Contexts}}{\text{Total Number of Contexts}}" />

The `HallucinationMetric` uses an LLM to determine, for each context in `contexts`, whether there are any contradictions to the `actual_output`.

:::info
Although extremely similar to the `FaithfulnessMetric`, the `HallucinationMetric` is calculated differently since it uses `contexts` as the source of truth instead. Since `contexts` is the ideal segment of your knowledge base relevant to a specific input, the degree of hallucination can be measured by the degree of which the `contexts` is disagreed upon.
:::
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)