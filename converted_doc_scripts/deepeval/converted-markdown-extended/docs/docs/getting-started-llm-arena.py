from deepeval import compare
from deepeval.metrics import ArenaGEval
from deepeval.test_case import ArenaTestCase, LLMTestCase
from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
from deepeval.test_case import LLMTestCaseParams
from jet.logger import logger
import NavigationCards from "@site/src/components/NavigationCards";
import os
import shutil
import { Timeline, TimelineItem } from "@site/src/components/Timeline";


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
id: getting-started-llm-arena
title: LLM Arena Evaluation
sidebar_label: LLM Arena
---


Learn how to evaluate different versions of your LLM app using LLM Arena-as-a-Judge in `deepeval`, a comparison-based LLM eval.

## Overview

Instead of comparing LLM outputs using a single-output LLM-as-a-Judge method as seen in previous sections, you can also compare n-pairwise test cases to find the best version of your LLM app. This method although does not provide numerical scores, allows you to more reliably choose the "winning" LLM output for a given set of inputs and outputs.

**In this 5 min quickstart, you'll learn how to:**

- Setup an LLM arena
- Use Arena G-Eval to pick the best performing LLM app

## Prerequisites

- Install `deepeval`

## Setup LLM Arena

In `deepeval`, arena test cases are used to compare different versions of your LLM app to see which one performs better. Each test case is an arena containing different contestants as different versions of your LLM app which are evaluated based on their corresponding `LLMTestCase`

<Timeline>

<TimelineItem title="Create an arena test case">

Create an `ArenaTestCase` by passing a dictionary of contestants with version names as keys and their corresponding `LLMTestCase` as values.
"""
logger.info("## Overview")


test_case = ArenaTestCase(
    contestants={
        "Version 1": LLMTestCase(
            input='Who wrote the novel "1984"?',
            actual_output="George Orwell",
        ),
        "Version 2": LLMTestCase(
            input='Who wrote the novel "1984"?',
            actual_output='"1984" was written by George Orwell.',
        ),
        "Version 3": LLMTestCase(
            input='Who wrote the novel "1984"?',
            actual_output="That dystopian masterpiece was penned by George Orwell 📚",
        ),
        "Version 4": LLMTestCase(
            input='Who wrote the novel "1984"?',
            actual_output="George Orwell is the brilliant mind behind the novel '1984'.",
        ),
    },
)

"""
You can learn more about `LLMTestCase` [here](https://deepeval.com/docs/evaluation-test-cases).

</TimelineItem>

<TimelineItem title="Define arena metric">

The [`ArenaGEval`](https://deepeval.com/docs/metrics-arena-g-eval) metric is the only metric that is compatible with `ArenaTestCase`. It picks a winner among the contestants based on the criteria defined.
"""
logger.info("You can learn more about `LLMTestCase` [here](https://deepeval.com/docs/evaluation-test-cases).")


arena_geval = ArenaGEval(
    name="Friendly",
    criteria="Choose the winner of the more friendly contestant based on the input and actual output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

"""
</TimelineItem>

</Timeline>

## Run Your First Arena Evals

Now that you have created an arena with contestants and defined a metric, you can run evals by using the `compare()` method:
"""
logger.info("## Run Your First Arena Evals")


test_case = ArenaTestCase(
    contestants={...}, # Use the same pairs you've created before
)

arena_geval = ArenaGEval(...) # Use the same metric you've created before

compare(test_cases=[test_case], metric=arena_geval)

"""
You can now run this python file to get your results:
"""
logger.info("You can now run this python file to get your results:")

python main.py

"""
This should let you see the results of the arena as shown below:

Counter({'Version 3': 1})

🎉🥳 **Congratulations!** You have just ran your first LLM arena-based evaluation. Here's what happened:

- When you call `compare()`, `deepeval` loops through each `ArenaTestCase`
- For each test case, `deepeval` uses the `ArenaGEval` metric to pick the "winner"
- To make the arena unbiased, `deepeval` masks the names of each contestant and randomizes their positions
- In the end, you get the number of "wins" each contestant got as the final output.

Unlike single-output LLM-as-a-Judge (which is everything but LLM arena evals), the concept of a "passing" test case does not exist for arena evals.

## Next Steps

Now that you have run your first Arena evals, you should:

1. **Customize your metrics**: You can change the criteria of your metric to be more specific to your use-case.
2. **Prepare a dataset**: If you don't have one, [generate one](/docs/synthesizer-introduction) as a starting point to store your inputs as goldens.

The arena metric is only used for picking winners among the contestants, it's not used for evaluating the answers themselves. To evaluate your LLM application on specific use cases you can read the other quickstarts here:

<NavigationCards
  columns={3}
  items={[
    {
      title: "AI Agents",
      icon: "Bot",
      listDescription: [
        "Setup LLM tracing",
        "Test end-to-end task completion",
        "Evaluate individual components",
      ],
      to: "/docs/getting-started-agents",
    },
    {
      title: "RAG",
      icon: "FileSearch",
      listDescription: [
        "Evaluate RAG end-to-end",
        "Test retriever and generator separately",
        "Multi-turn RAG evals",
      ],
      to: "/docs/getting-started-rag",
    },
    {
      title: "Chatbots",
      icon: "MessagesSquare",
      listDescription: [
        "Setup multi-turn test cases",
        "Evaluate turns in a conversation",
        "Simulate user interactions",
      ],
      to: "/docs/getting-started-chatbots",
    },
  ]}
/>
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)