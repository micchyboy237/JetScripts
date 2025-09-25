from deepeval.metrics import ArenaTestCase, LLMTestCaseParams
from deepeval.test_case import ArenaTestCase, LLMTestCase
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
id: evaluation-arena-test-cases
title: Arena Test Case
sidebar_label: Arena
---

## Quick Summary

An **arena test case** is a blueprint provided by `deepeval` for you to compare which iteration of your LLM app performed better. It works by comparing each contestants's `LLMTestCase` to run comparisons, and currently only supports the `LLMTestCase` for single-turn, text-based comparisons.

:::info
Support for `ConversationalTestCase` and `MLLMTestCase` is coming soon.
:::

The `ArenaTestCase` currently only runs with the `ArenaGEval` metric, and all that is required is to provide a dictionary of contestant names to test cases:
"""
logger.info("## Quick Summary")


test_case = ArenaTestCase(
    contestants={
        "GPT-4": LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
        ),
        "Claude-4": LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
        ),
        "Gemini 2.0": LLMTestCase(
            input="What is the capital of France?",
            actual_output="Absolutely! The capital of France is Paris 😊",
        ),
        "Deepseek R1": LLMTestCase(
            input="What is the capital of France?",
            actual_output="Hey there! It’s Paris—the beautiful City of Light. Have a wonderful day!",
        ),
    },
)

"""
Note that all `input`s and `expected_output`s you provide across contestants **MUST** match.

:::tip
For those wondering why we took the choice to include multiple duplicated `input`s in `LLMTestCase` instead of moving it to the `ArenaTestCase` class, it is because an `LLMTestCase` integrates nicely with the existing ecosystem.

You also shouldn't worry about unexpected errors because `deepeval` will throw an error if `input`s or `expected_output`s aren't matching.
:::

## Arena Test Case

The `ArenaTestCase` takes a simple `contestants` argument, which is a dictionary of contestant names to `LLMTestCase`s:
"""
logger.info("## Arena Test Case")

contestants = {
    "GPT-4": LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
    ),
    "Claude-4": LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
    ),
    "Gemini 2.0": LLMTestCase(
        input="What is the capital of France?",
        actual_output="Absolutely! The capital of France is Paris 😊",
    ),
    "Deepseek R1": LLMTestCase(
        input="What is the capital of France?",
        actual_output="Hey there! It’s Paris—the beautiful City of Light. Have a wonderful day!",
    ),
}

test_case = ArenaTestCase(contestants=contestants)

"""
The [`ArenaGEval` metric](/docs/metrics-arena-g-eval) is the only metric that uses an `ArenaTestCase`, which pickes a "winner" out of the list of contestants:
"""
logger.info("The [`ArenaGEval` metric](/docs/metrics-arena-g-eval) is the only metric that uses an `ArenaTestCase`, which pickes a "winner" out of the list of contestants:")

...

arena_geval = ArenaGEval(
    name="Friendly",
    criteria="Choose the winner of the more friendly contestant based on the input and actual output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

arena_geval.measure(test_case)
logger.debug(arena_geval.winner, arena_geval.reason)

"""
The `ArenaTestCase` streamlines the evaluation by automatically masking contestant names (to ensure unbiased judging) and randomizing their order.
"""
logger.info("The `ArenaTestCase` streamlines the evaluation by automatically masking contestant names (to ensure unbiased judging) and randomizing their order.")

logger.info("\n\n[DONE]", bright=True)