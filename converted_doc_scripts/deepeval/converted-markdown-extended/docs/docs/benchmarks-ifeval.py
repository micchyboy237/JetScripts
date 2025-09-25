from deepeval.benchmarks import IFEval
from jet.logger import logger
import Equation from "@site/src/components/Equation";
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
id: benchmarks-ifeval
title: IFEval
sidebar_label: IFEval
---


**IFEval (Instruction-Following Evaluation for Large Language Models
)** is a benchmark for evaluating instruction-following capabilities of language models.
It tests various aspects of instruction following including format compliance, constraint
adherence, output structure requirements, and specific instruction types.

:::tip
`deepeval`'s `IFEval` implementation is based on the [original research paper](https://arxiv.org/abs/2311.07911) by Google.
:::

## Arguments

There is **ONE** optional argument when using the `IFEval` benchmark:

- [Optional] `n_problems`: limits the number of test cases the benchmark will evaluate. Defaulted to `None`.

## Usage

The code below evaluates a custom `mistral_7b` model ([click here to learn how to use **ANY** custom LLM](/docs/benchmarks-introduction#benchmarking-your-llm)) and assesses its performance on High School Computer Science and Astronomy using 3-shot learning.
"""
logger.info("## Arguments")


benchmark = IFEval(n_problems=5)

benchmark.evaluate(model=mistral_7b)
logger.debug(benchmark.overall_score)

logger.info("\n\n[DONE]", bright=True)