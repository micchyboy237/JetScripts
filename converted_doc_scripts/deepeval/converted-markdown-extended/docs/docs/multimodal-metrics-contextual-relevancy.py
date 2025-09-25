from deepeval import evaluate
from deepeval.metrics import MultimodalContextualRelevancyMetric
from deepeval.test_case import MLLMTestCase, MLLMImage
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
id: multimodal-metrics-contextual-relevancy
title: Multimodal Contextual Relevancy
sidebar_label: Multimodal Contextual Relevancy
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/multimodal-metrics-contextual-relevancy"
  />
</head>


<MetricTagsDisplayer singleTurn={true} custom={true} multimodal={true} />

The multimodal contextual relevancy metric measures the quality of your multimodal RAG pipeline's retriever by evaluating the overall relevance of the information presented in your `retrieval_context` for a given `input`. `deepeval`'s multimodal contextual relevancy metric is a self-explaining MLLM-Eval, meaning it outputs a reason for its metric score.

:::info
The **Multimodal Contextual Relevancy** is the multimodal adaptation of DeepEval's [contextual relevancy metric](/docs/metrics-contextual-relevancy). It accepts images in addition to text for the `input`, `actual_output`, and `retrieval_context`.
:::

## Required Arguments

To use the `MultimodalContextualRelevancyMetric`, you'll have to provide the following arguments when creating a [`MLLMTestCase`](/docs/evaluation-test-cases#mllm-test-case):

- `input`
- `actual_output`
- `retrieval_context`

:::note
Similar to `MultimodalContextualPrecisionMetric`, the `MultimodalContextualRelevancyMetric` uses `retrieval_context` from your multimodal RAG pipeline for evaluation.
:::

The `input` and `actual_output` are required to create an `MLLMTestCase` (and hence required by all metrics) even though they might not be used for metric calculation. Read the [How Is It Calculated](#how-is-it-calculated) section below to learn more.

## Usage
"""
logger.info("## Required Arguments")


metric = MultimodalContextualRelevancyMetric()
m_test_case = MLLMTestCase(
    input=["Tell me about some landmarks in France"],
    actual_output=[
        "France is home to iconic landmarks like the Eiffel Tower in Paris.",
        MLLMImage(...)
    ],
    retrieval_context=[
        MLLMImage(...),
        "The Eiffel Tower is a wrought-iron lattice tower built in the late 19th century.",
        MLLMImage(...)
    ],
)


evaluate(test_case=[m_test_case], metrics=[metric])

"""
There are **SIX** optional parameters when creating a `MultimodalContextualRelevancyMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's Multimodal GPT models to use, **OR** any custom MLLM model of type `DeepEvalBaseMLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

### As a standalone

You can also run the `MultimodalContextualRelevancyMetric` on a single test case as a standalone, one-off execution.
"""
logger.info("### As a standalone")

...

metric.measure(m_test_case)
logger.debug(metric.score, metric.reason)

"""
## How Is It Calculated?

The `MultimodalContextualRelevancyMetric` score is calculated according to the following equation:

<Equation formula="\text{Multimodal Contextual Relevancy} = \frac{\text{Number of Relevant Statements}}{\text{Total Number of Statements}}" />

Although similar to how the `MultimodalAnswerRelevancyMetric` is calculated, the `MultimodalContextualRelevancyMetric` first uses an MLLM to extract all statements and images in the `retrieval_context` instead, before using the same MLLM to classify whether each statement and image is relevant to the `input`.
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)