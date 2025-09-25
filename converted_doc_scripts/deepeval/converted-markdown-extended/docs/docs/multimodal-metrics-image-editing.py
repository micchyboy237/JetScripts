from deepeval import evaluate
from deepeval.metrics import ImageEditingMetric
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
id: multimodal-metrics-image-editing
title: Image Editing
sidebar_label: Image Editing
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/multimodal-metrics-image-editing"
  />
</head>


<MetricTagsDisplayer singleTurn={true} custom={true} multimodal={true} />

The Image Editing metric assesses the performance of **image editing tasks** by evaluating the quality of synthesized images based on semantic consistency and perceptual quality (similar to the `TextToImageMetric`). `deepeval`'s Image Editing metric is a self-explaining MLLM-Eval, meaning it outputs a reason for its metric score.

## Required Arguments

To use the `ImageEditingMetric`, you'll have to provide the following arguments when creating a [`MLLMTestCase`](/docs/evaluation-test-cases#mllm-test-case):

- `input`
- `actual_output`

:::note
Both the input and output should each contain exactly **1 image**.
:::

The `input` and `actual_output` are required to create an `MLLMTestCase` (and hence required by all metrics) even though they might not be used for metric calculation. Read the [How Is It Calculated](#how-is-it-calculated) section below to learn more.

## Usage
"""
logger.info("## Required Arguments")


metric = ImageEditingMetric(
    threshold=0.7,
    include_reason=True,
)
m_test_case = MLLMTestCase(
    input=["Change the color of the shoes to blue.", MLLMImage(url="./shoes.png", local=True)],
    actual_output=[MLLMImage(url="https://shoe-images.com/edited-shoes", local=False)]
)


evaluate(test_cases=[m_test_case], metrics=[metric])

"""
There are **FIVE** optional parameters when creating a `ImageEditingMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

### As a standalone

You can also run the `ImageEditingMetric` on a single test case as a standalone, one-off execution.
"""
logger.info("### As a standalone")

...

metric.measure(m_test_case)
logger.debug(metric.score, metric.reason)

"""
## How Is It Calculated?

The `ImageEditingMetric` score is calculated according to the following equation:

<Equation formula="O = \sqrt{\text{min}(\alpha_1, \ldots, \alpha_i) \cdot \text{min}(\beta_1, \ldots, \beta_i)}" />

The `ImageEditingMetric` score combines Semantic Consistency (SC) and Perceptual Quality (PQ) sub-scores to provide a comprehensive evaluation of the synthesized image. The final overall score is derived by taking the square root of the product of the minimum SC and PQ scores.

### SC Scores

These scores assess aspects such as alignment with the prompt and resemblance to concepts. The minimum value among these sub-scores represents the SC score. During the SC evaluation, both the input conditions and the synthesized image are used.

### PQ Scores

These scores evaluate the naturalness and absence of artifacts in the image. The minimum value among these sub-scores represents the PQ score. For the PQ evaluation, only the synthesized image is used to prevent confusion from the input conditions.
"""
logger.info("## How Is It Calculated?")

logger.info("\n\n[DONE]", bright=True)