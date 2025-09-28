from deepeval.benchmarks import ARC
from deepeval.benchmarks.modes import ARCMode
from jet.adapters.haystack.deepeval.ollama_model import OllamaModel
from jet.logger import logger
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
id: benchmarks-arc
title: ARC
sidebar_label: ARC
---

<head>
  <link rel="canonical" href="https://deepeval.com/docs/benchmarks-arc" />
</head>

**ARC or AI2 Reasoning Challenge** is a dataset used to benchmark language models' reasoning abilities. The benchmark consists of 8,000 multiple-choice questions from science exams for grades 3 to 9. The dataset includes two modes: _easy_ and _challenge_, with the latter featuring more difficult questions that require advanced reasoning.

:::tip
To learn more about the dataset and its construction, you can [read the original paper here](https://arxiv.org/pdf/1803.05457v1).
:::

## Arguments

There are **THREE** optional arguments when using the `ARC` benchmark:

- [Optional] `n_problems`: the number of problems for model evaluation. By default, this is set all problems available in each benchmark mode.
- [Optional] `n_shots`: the number of examples for few-shot learning. This is **set to 5** by default and **cannot exceed 5**.
- [Optional] mode: a `ARCMode` enum that selects the evaluation mode. This is set to `ARCMode.EASY` by default. `deepeval` currently supports 2 modes: **EASY and CHALLENGE**.

:::info
Both `EASY` and `CHALLENGE` modes consist of **multiple-choice** questions. However, `CHALLENGE` questions are more difficult and require more advanced reasoning.
:::

## Usage

The code below assesses a custom `mistral_7b` model ([click here to learn how to use **ANY** custom LLM](/docs/benchmarks-introduction#benchmarking-your-llm)) on 100 problems in `ARC` in EASY mode.
"""
logger.info("## Arguments")


benchmark = ARC(
    n_problems=100,
    n_shots=3,
    mode=ARCMode.EASY
)

model = OllamaModel(
    model="deepseek-r1:1.5b-qwen-distill-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0
)

benchmark.evaluate(model=model)
logger.debug(benchmark.overall_score)

"""
The `overall_score` ranges from 0 to 1, signifying the fraction of accurate predictions across tasks. Both modes' performances are measured using an **exact match** scorer, focusing on the quantity of correct answers.
"""
logger.info("The `overall_score` ranges from 0 to 1, signifying the fraction of accurate predictions across tasks. Both modes' performances are measured using an **exact match** scorer, focusing on the quantity of correct answers.")

logger.info("\n\n[DONE]", bright=True)