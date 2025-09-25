from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.metrics import JsonCorrectnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from jet.logger import logger
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
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
id: metrics-json-correctness
title: Json Correctness
sidebar_label: Json Correctness
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-json-correctness"
  />
</head>


<MetricTagsDisplayer singleTurn={true} usesLLMs={false} referenceless={true} />

The json correctness metric measures whether your LLM application is able to generate `actual_output`s with the correct **json schema**.

:::note

The `JsonCorrectnessMetric` like the `ToolCorrectnessMetric` is not an LLM-eval, and you'll have to supply your expected Json schema when creating a `JsonCorrectnessMetric`.

:::

## Required Arguments

To use the `JsonCorrectnessMetric`, you'll have to provide the following arguments when creating an [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case):

- `input`
- `actual_output`

Read the [How Is It Calculated](#how-is-it-calculated) section below to learn how test case parameters are used for metric calculation.

## Usage

First define your schema by creating a `pydantic` `BaseModel`:
"""
logger.info("## Required Arguments")


class ExampleSchema(BaseModel):
    name: str

"""
:::tip
If your `actual_output` is a list of JSON objects, you can simply create a list schema by wrapping your existing schema in a `RootModel`. For example:
"""
logger.info("If your `actual_output` is a list of JSON objects, you can simply create a list schema by wrapping your existing schema in a `RootModel`. For example:")


...

class ExampleSchemaList(RootModel[List[ExampleSchema]]):
    pass

"""
:::

Then supply it as the `expected_schema` when creating a `JsonCorrectnessMetric`, which can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluation:
"""
logger.info("Then supply it as the `expected_schema` when creating a `JsonCorrectnessMetric`, which can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluation:")



metric = JsonCorrectnessMetric(
    expected_schema=ExampleSchema,
    model="llama3.2",
    include_reason=True
)
test_case = LLMTestCase(
    input="Output me a random Json with the 'name' key",
    actual_output="{'name': null}"
)


evaluate(test_cases=[test_case], metrics=[metric])

"""
There are **ONE** mandatory and **SIX** optional parameters when creating an `PromptAlignmentMetric`:

- `expected_schema`: a `pydantic` `BaseModel` specifying the schema of the Json that is expected from your LLM.
- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use to generate reasons, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-a-metric-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

:::info
Unlike other metrics, the `model` is used for generating reason instead of evaluation. It will only be used if the `actual_output` has the wrong schema, **AND** if `include_reason` is set to `True`.
:::

### Within components

You can also run the `JsonCorrectnessMetric` within nested components for [component-level](/docs/evaluation-component-level-llm-evals) evaluation.
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

You can also run the `JsonCorrectnessMetric` on a single test case as a standalone, one-off execution.
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

<Equation
  formula="\text{Json Correctness} = \begin{cases}
1 & \text{If the actual output fits the expected schema}, \\
0 & \text{Otherwise}
\end{cases}"
/>

The `JsonCorrectnessMetric` does not use an LLM for evaluation and instead uses the provided `expected_schema` to determine whether the `actual_output` can be loaded into the schema.
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)