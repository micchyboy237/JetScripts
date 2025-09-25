from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe
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
id: metrics-task-completion
title: Task Completion
sidebar_label: Task Completion
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/metrics-task-completion"
  />
</head>


<MetricTagsDisplayer singleTurn={true} agent={true} referenceless={true} />

The task completion metric uses LLM-as-a-judge to evaluate how effectively an **LLM agent accomplishes a task**. Task Completion is a self-explaining LLM-Eval, meaning it outputs a reason for its metric score.

:::info
Task Completion analyzes your **agent's full trace** to determine task success, which requires [setting up tracing](/docs/evaluation-llm-tracing).
:::

## Usage

To begin, [set up tracing](/docs/evaluation-llm-tracing) and simply supply the `TaskCompletionMetric()` to your agent's `@observe` tag.
"""
logger.info("## Usage")


@observe()
def trip_planner_agent(input):
    destination = "Paris"
    days = 2

    @observe()
    def restaurant_finder(city):
        return ["Le Jules Verne", "Angelina Paris", "Septime"]

    @observe()
    def itinerary_generator(destination, days):
        return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return itinerary + restaurants


dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

task_completion = TaskCompletionMetric(threshold=0.7, model="llama3.2")

for goldens in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)

"""
There are **SEVEN** optional parameters when creating a `TaskCompletionMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `task`: a string representing the task to be completed. If no task is supplied, it is automatically inferred from the trace. Defaulted to the `None`
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4o'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-a-metric-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

To learn more about how the `evals_iterator` work, [click here.](/docs/evaluation-end-to-end-llm-evals#e2e-evals-for-tracing)

## How Is It Calculated?

The `TaskCompletionMetric` score is calculated according to the following equation:

<Equation formula="\text{Task Completion Score} = \text{AlignmentScore}(\text{Task}, \text{Outcome})" />

- **Task** and **Outcome** are extracted from the trace (or test case for end-to-end) using an LLM.
- The **Alignment Score** measures how well the outcome aligns with the extracted (or user-provided) task, as judged by an LLM.
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)