from deepeval import evaluate
from deepeval.metrics import RoleAdherenceMetric
from deepeval.test_case import Turn, ConversationalTestCase
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
id: metrics-role-adherence
title: Role Adherence
sidebar_label: Role Adherence
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-role-adherence"
  />
</head>


<MetricTagsDisplayer multiTurn={true} chatbot={true} referenceless={true} />

The role adherence metric is a conversational metric that determines whether your LLM chatbot is able to adhere to its given role **throughout a conversation**.

:::tip
The `RoleAdherenceMetric` is particularly useful for a role-playing use case.
:::

## Required Arguments

To use the `RoleAdherenceMetric`, you'll have to provide the following arguments when creating a [`ConversationalTestCase`](/docs/evaluation-multiturn-test-cases):

- `turns`
- `chatbot_role`

You must provide the `role` and `content` for evaluation to happen. Read the [How Is It Calculated](#how-is-it-calculated) section below to learn more.

## Usage

The `RoleAdherenceMetric()` can be used for [end-to-end](/docs/evaluation-end-to-end-llm-evals) multi-turn evaluation:
"""
logger.info("## Required Arguments")


convo_test_case = ConversationalTestCase(
    chatbot_role="...",
    turns=[Turn(role="...", content="..."), Turn(role="...", content="...")]
)
metric = RoleAdherenceMetric(threshold=0.5)


evaluate(test_cases=[convo_test_case], metrics=[metric])

"""
There are **SIX** optional parameters when creating a `RoleAdherenceMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

### As a standalone

You can also run the `RoleAdherenceMetric` on a single test case as a standalone, one-off execution.
"""
logger.info("### As a standalone")

...

metric.measure(convo_test_case)
logger.debug(metric.score, metric.reason)

"""
:::caution
This is great for debugging or if you wish to build your own evaluation pipeline, but you will **NOT** get the benefits (testing reports, Confident AI platform) and all the optimizations (speed, caching, computation) the `evaluate()` function or `deepeval test run` offers.
:::

## How Is It Calculated?

The `RoleAdherenceMetric` score is calculated according to the following equation:

<Equation formula="\text{Role Adherence} = \frac{\text{Number of Assistant Turns that Adhered to Chatbot Role in Conversation}}{\text{Total Number of Assistant Turns in Conversation}}" />

The `RoleAdherenceMetric` iterates over each assistant turn and uses an LLM to evaluate whether the content adheres to the specified `chatbot_role`, using previous conversation turns as context.
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)