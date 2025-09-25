from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent
from deepeval.metrics import AnswerRelevancyMetric
from jet.logger import logger
from pydantic_ai import Agent
import ColabButton from "@site/src/components/ColabButton";
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import asyncio
import os
import shutil
import time
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
id: pydanticai
title: Pydantic AI
sidebar_label: Pydantic AI
---


# Pydantic AI

<ColabButton 
  notebookUrl="https://colab.research.google.com/github/confident-ai/deepeval/blob/main/cookbooks/examples/notebooks/pydantic_ai.ipynb" 
  className="header-colab-button"
/>

[Pydantic AI](https://ai.pydantic.dev/) is a Python framework for building reliable, production-grade applications with Generative AI, providing type safety and validation for agent outputs and LLM interactions.

:::tip
We recommend logging in to [Confident AI](https://app.confident-ai.com) to view your Pydantic AI evaluations.
"""
logger.info("# Pydantic AI")

deepeval login

"""
:::

## End-to-End Evals

`deepeval` allows you to evaluate Pydantic AI applications end-to-end in **under a minute**.

<Timeline>

<TimelineItem title="Configure Pydantic AI">

Create agent and pass `metrics` to the `deepeval`'s `Agent` wrapper.
"""
logger.info("## End-to-End Evals")


instrument_pydantic_ai()

agent = Agent(
    "ollama:llama3.2",
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync("What are the LLMs?")
logger.debug(result)
time.sleep(10) # wait for the trace to be posted

"""
:::info
Evaluations are supported for Pydantic AI `Agent`. Only metrics with parameters `input` and `output` are eligible for evaluation.
:::

</TimelineItem>
<TimelineItem title="Run evaluations">

Create an `EvaluationDataset` and invoke your Pydantic AI application for each golden within the `evals_iterator()` loop to run end-to-end evaluations.

<Tabs groupId="pydantic_ai">
<TabItem value="asynchronous" label="Asynchronous">
"""
logger.info("Evaluations are supported for Pydantic AI `Agent`. Only metrics with parameters `input` and `output` are eligible for evaluation.")


instrument_pydantic_ai()
agent = Agent("ollama:llama3.2", system_prompt="Be concise, reply with one sentence.")
answer_relavancy_metric = AnswerRelevancyMetric()


dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
        Golden(input="What's 7 * 6?"),
    ]
)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(agent.run(
        golden.input,
        metrics=[answer_relavancy_metric],
    ))
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

✅ Done. The `evals_iterator` will automatically generate a test run with individual evaluation traces for each golden.

</TimelineItem>
<TimelineItem title="View on Confident AI (optional)">

<VideoDisplayer
  src="https://confident-bucket.s3.us-east-1.amazonaws.com/end-to-end%3Apydantic-1080.mp4"
/>

</TimelineItem>

</Timeline>

:::note
If you need to evaluate individual components of your Pydantic AI application, [set up tracing](/docs/evaluation-llm-tracing) instead.
:::

## Component-level Evals

DeepEval supports evaluating individual components of your Pydantic AI application.

### Tool
Pass `metrics` or `metric_collection` to the `tool` decorator from DeepEval's Pydantic AI `Agent`.
"""
logger.info("## Component-level Evals")


weather_agent = Agent(
    "ollama:llama3.2",
    instructions='Be concise, reply with one sentence.'
)

@weather_agent.tool(metric_collection="test_collection_1")
async def get_lat_lng(location_description: str) -> (float, float):
    return 1.0, 2.0

"""
## Evals in Production

To run online evaluations in production, replace `metrics` with a [metric collection](https://documentation.confident-ai.com/docs/llm-tracing/evaluations#online-evaluations) string from Confident AI, and push your Pydantic AI agent to production.
"""
logger.info("## Evals in Production")


instrument_pydantic_ai()

agent = Agent(
    "ollama:llama3.2",
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync(
    "What are the LLMs?",
    metric_collection="test_collection_1",
)

logger.debug(result)
time.sleep(10) # wait for the trace to be posted

logger.info("\n\n[DONE]", bright=True)