from crewai import Task, Crew
from deepeval.integrations.crewai import Agent
from deepeval.integrations.crewai import instrument_crewai
from deepeval.metrics import AnswerRelevancyMetric
from jet.logger import logger
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
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
id: crewai
title: CrewAI
sidebar_label: CrewAI
---


# CrewAI

<ColabButton 
  notebookUrl="https://colab.research.google.com/github/confident-ai/deepeval/blob/main/cookbooks/examples/notebooks/crewai.ipynb" 
  className="header-colab-button"
/>

[CrewAI](https://www.crewai.com/) is a lean, independent Python framework designed for creating and orchestrating autonomous multi-agent AI systems, offering high flexibility, speed, and precision control for complex automation tasks.

:::tip
We recommend logging in to [Confident AI](https://app.confident-ai.com) to view your CrewAI evaluation traces.
"""
logger.info("# CrewAI")

deepeval login

"""
:::

## End-to-End Evals

`deepeval` allows you to evaluate CrewAI applications end-to-end in **under a minute**.

<Timeline>

<TimelineItem title="Configure CrewAI">

Create a `Crew` and pass `metrics` to the `deepeval`'s `Agent` wrapper.
"""
logger.info("## End-to-End Evals")



instrument_crewai()

answer_relavancy_metric = AnswerRelevancyMetric()

agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metrics=[answer_relavancy_metric]
)

task = Task(
    description="Explain the given topic",
    expected_output="A clear and concise explanation.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

"""
:::info
Evaluations are supported for CrewAI `Agent`. Only metrics with parameters `input`, `output`, `expected_output` and `tools_called` are eligible for evaluation.
:::

</TimelineItem>
<TimelineItem title="Run evaluations">

Create an `EvaluationDataset` and invoke your CrewAI application for each golden within the `evals_iterator()` loop to run end-to-end evaluations.

<Tabs groupId="crewai">
<TabItem value="synchronous" label="Synchronous">
"""
logger.info("Evaluations are supported for CrewAI `Agent`. Only metrics with parameters `input`, `output`, `expected_output` and `tools_called` are eligible for evaluation.")

dataset = EvaluationDataset(goldens=[
    Golden(input="What are Transformers in AI?"),
    Golden(input="What is the biggest open source database?"),
    Golden(input="What are LLMs?"),
])

for golden in dataset.evals_iterator():
    result = crew.kickoff(inputs={"input": golden.input})

"""
</TabItem>
  <TabItem value="asynchronous" label="Asynchronous">
"""

dataset = EvaluationDataset(goldens=[
    Golden(input="What are Transformers in AI?"),
    Golden(input="What is the biggest open source database?"),
    Golden(input="What are LLMs?"),
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(crew.kickoff_async(inputs={"input": golden.input}))
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

✅ Done. The `evals_iterator` will automatically generate a test run with individual evaluation traces for each golden.

</TimelineItem>
<TimelineItem title="View on Confident AI (optional)">

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/end-to-end%3Acrewai-4k-no-zoom.mp4"
/>

</TimelineItem>

</Timeline>

:::note
If you need to evaluate individual components of your CrewAI application, [set up tracing](/docs/evaluation-llm-tracing) instead.
:::

## Evals in Production

To run online evaluations in production, replace `metrics` with a [metric collection](https://documentation.confident-ai.com/docs/llm-tracing/evaluations#online-evaluations) string from Confident AI, and push your CrewAI agent to production.
"""
logger.info("## Evals in Production")

...
agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metric_collection="test_collection_1",
)

result = crew.kickoff(
    "input": "What are the LLMs?"
)

logger.info("\n\n[DONE]", bright=True)