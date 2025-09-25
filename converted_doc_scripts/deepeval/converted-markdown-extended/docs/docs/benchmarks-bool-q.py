from deepeval.benchmarks import BoolQ
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
id: benchmarks-bool-q
title: BoolQ
sidebar_label: BoolQ
---

<head>
  <link rel="canonical" href="https://deepeval.com/docs/benchmarks-bool-q" />
</head>

**BoolQ** is a reading comprehension dataset containing 16K yes/no questions (3.3K in the validation set). BoolQ features naturally occurring questions, meaning they are generated in an unprompted setting, with each question accompanied by a passage.

:::info
To learn more about the dataset and its construction, you can [read the original paper here](https://arxiv.org/pdf/1905.10044).
:::

## Arguments

There are **TWO** optional arguments when using the `BoolQ` benchmark:

- [Optional] `n_problems`: the number of problems for model evaluation. By default, this is set to 3270 (all problems).
- [Optional] `n_shots`: the number of examples for few-shot learning. This is **set to 5** by default and **cannot exceed 5**.

## Usage

The code below assesses a custom `mistral_7b` model ([click here to learn how to use **ANY** custom LLM](/docs/benchmarks-introduction#benchmarking-your-llm)) on 10 problems in `BoolQ` using 3-shot CoT prompting.
"""
logger.info("## Arguments")


benchmark = BoolQ(
    n_problems=10,
    n_shots=3,
)

benchmark.evaluate(model=mistral_7b)
logger.debug(benchmark.overall_score)

"""
The `overall_score` for this benchmark ranges from 0 to 1, where 1 signifies perfect performance and 0 indicates no correct answers. The model's score, based on **exact matching**, is calculated by determining the proportion of questions for which the model produces the precise correct answer (i.e. 'Yes' or 'No') in relation to the total number of questions.

:::tip
As a result, utilizing more few-shot prompts (`n_shots`) can greatly improve the model's robustness in generating answers in the exact correct format and boost the overall score.
:::
"""
logger.info("The `overall_score` for this benchmark ranges from 0 to 1, where 1 signifies perfect performance and 0 indicates no correct answers. The model's score, based on **exact matching**, is calculated by determining the proportion of questions for which the model produces the precise correct answer (i.e. 'Yes' or 'No') in relation to the total number of questions.")

logger.info("\n\n[DONE]", bright=True)