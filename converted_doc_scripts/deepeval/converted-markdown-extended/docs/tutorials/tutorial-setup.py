from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
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
id: tutorial-setup
title: Set Up DeepEval
sidebar_label: Set Up DeepEval
---

## Installing DeepEval

**DeepEval** is a powerful LLM evaluation framework. Here's how you can easily get started by installing and running your first evaluation using DeepEval.

Start by installing DeepEval using pip:
"""
logger.info("## Installing DeepEval")

pip install -U deepeval

"""
### Write your first test

Let's evaluate the correctness of an LLM output using [`GEval`](https://deepeval.com/docs/metrics-llm-evals), a powerful metric based on LLM-as-a-judge evaluation.

:::note
Your test file must be named with a `test_` prefix (like `test_app.py`) for DeepEval to recognize and run it.
:::
"""
logger.info("### Write your first test")


correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
)

evaluate([test_case], [correctness_metric])

"""
To run your first evaluation, enter the following command in your terminal:
"""
logger.info("To run your first evaluation, enter the following command in your terminal:")

deepeval test run test_app.py

"""
:::note
DeepEval's powerful **LLM-as-a-judge** metrics (like `GEval` used in this example) rely on an underlying LLM called the _Evaluation Model_ to perform evaluations. By default, DeepEval uses Ollama's models for this purpose.

# So you'll have to set your `OPENAI_API_KEY` as an environment variable as shown below.
"""
logger.info("DeepEval's powerful **LLM-as-a-judge** metrics (like `GEval` used in this example) rely on an underlying LLM called the _Evaluation Model_ to perform evaluations. By default, DeepEval uses Ollama's models for this purpose.")

# export OPENAI_API_KEY="your_api_key"

"""
To use ANY custom LLM of your choice, [Check out our docs on custom evaluation models](https://deepeval.com/guides/guides-using-custom-llms).
:::

Congratulations! You've successfully run your first LLM evaluation with DeepEval.

## Setting Up Confident AI

While DeepEval works great standalone, you can connect it to [Confident AI](https://www.confident-ai.com) — our cloud platform for dashboards, logging, collaboration, and more — built for LLM evaluation. **It’s free to get started.**

You can [sign up here](https://www.confident-ai.com), or run:
"""
logger.info("## Setting Up Confident AI")

deepeval login

"""
Navigate to your Settings page and copy your Confident AI API Key from the Project API Key box. If you used the `deepeval login` command to log in, you'll be prompted to paste your Confident AI API Key after creating an account.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  }}
>
  <img
    src="https://deepeval-docs.s3.amazonaws.com/tutorial_setup_01.svg"
    style={{
      marginTop: "20px",
      marginBottom: "20px",
      height: "auto",
      maxHeight: "800px",
    }}
  />
</div>

Alternatively, if you already have an account, you can log in directly using Python:
"""
logger.info("Navigate to your Settings page and copy your Confident AI API Key from the Project API Key box. If you used the `deepeval login` command to log in, you'll be prompted to paste your Confident AI API Key after creating an account.")

deepeval.login("your-confident-api-key")

"""
Or through the CLI:
"""
logger.info("Or through the CLI:")

deepeval login --confident-api-key "your-confident-api-key"

"""
:::note Login persistence
`deepeval login` persists your key to a dotenv file by default (.env.local).
To change the target, use `--save`, e.g.:
"""
logger.info("To change the target, use `--save`, e.g.:")

deepeval login --confident-api-key "ck_..." --save dotenv:.env.custom

"""
For compatibility, the key is saved under `api_key` and `CONFIDENT_API_KEY`.
Secrets are never written to the JSON keystore.
:::

:::tip Logging out / rotating keys
Use deepeval logout to clear the JSON keystore and remove saved keys from your dotenv file:
"""
logger.info("For compatibility, the key is saved under `api_key` and `CONFIDENT_API_KEY`.")

deepeval logout

deepeval logout --save dotenv:.myconf.env

"""
:::

You're all set! You can now evaluate LLMs locally and monitor them in Confident AI.
"""
logger.info("You're all set! You can now evaluate LLMs locally and monitor them in Confident AI.")

logger.info("\n\n[DONE]", bright=True)