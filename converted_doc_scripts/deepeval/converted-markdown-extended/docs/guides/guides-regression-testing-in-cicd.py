from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from jet.logger import logger
import os
import pytest
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
# id: guides-regression-testing-in-cicd
title: Regression Testing LLM Systems in CI/CD
sidebar_label: Regression Testing in CI/CD
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/guides/guides-regression-testing-in-cicd"
  />
</head>

Regression testing ensures your LLM systems doesn't degrade in performance over time, and there is no better place to do it than in CI/CD environments. `deepeval` allows anyone to easily regression test outputs of LLM systems (which can be RAG pipelines, or even just an LLM itself) in the CLI through its deep integration with Pytest via the `deepeval test run` command.

:::info
This guide will show how you can include `deepeval` in your CI/CD pipelines, using GitHub Actions as an example.
:::

## Creating Your Test File

`deepeval` treats rows in an evaluation dataset as unit test cases, and a wide range of research backed LLM evaluation metrics, which you can define in a `test_<name>.py` file to implement your regression test.
"""
logger.info("# id: guides-regression-testing-in-cicd")


first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")
dataset = EvaluationDataset(
    test_cases=[first_test_case, second_test_case]
)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_example(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])

"""
:::tip
In the example shown above, the `LLMTestCase`s are hardcoded for demonstration purposes only. Instead, you should aim to choose one of the [three ways `deepeval` offers to load a dataset](/docs/evaluation-datasets#load-an-existing-dataset) in a more scalable way.
:::

To check that your test file is working, run `deepeval test run`:
"""
logger.info("In the example shown above, the `LLMTestCase`s are hardcoded for demonstration purposes only. Instead, you should aim to choose one of the [three ways `deepeval` offers to load a dataset](/docs/evaluation-datasets#load-an-existing-dataset) in a more scalable way.")

deepeval test run test_file.py

"""
## Setting Up Your YAML File

To set up a GitHub workflow that triggers `deepeval test run` on every pull or push request, define a `.yaml` file:
"""
logger.info("## Setting Up Your YAML File")

name: LLM Regression Test

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install Dependencies
        run: poetry install --no-root

      - name: Run DeepEval Unit Tests
        run: poetry run deepeval test run test_file.py

"""
**Congratulations 🎉!** You've now setup an automated regression testing suite in under 30 lines of code.

:::note
Although we only showed GitHub workflows in this guide, it will be extremely similar even if you're using another CI/CD environment such as Travis CI or CircleCI.

# You should also note that you don't have to strictly use poetry (as shown in the example above) to install dependencies, and you may need to configure additional environment variables such as an `OPENAI_API_KEY` if you're using GPT models for evaluation and a `CONFIDENT_API_KEY` if you're using Confident AI to keep track of testing results.
:::
"""
logger.info("Although we only showed GitHub workflows in this guide, it will be extremely similar even if you're using another CI/CD environment such as Travis CI or CircleCI.")

logger.info("\n\n[DONE]", bright=True)