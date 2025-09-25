from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.llama_index import instrument_llama_index, FunctionAgent
from deepeval.metrics import AnswerRelevancyMetric
from jet.adapters.llama_index.ollama_function_calling import Ollama
from jet.logger import logger
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import asyncio
import llama_index.core.instrumentation as instrument
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
id: llamaindex
title: LlamaIndex
sidebar_label: LlamaIndex
---



# LlamaIndex

[LlamaIndex](https://www.llamaindex.ai/) is an orchestration framework that simplifies data ingestion, indexing, and querying, allowing developers to integrate private and public data into LLM applications for retrieval-augmented generation and knowledge augmentation.

:::tip
We recommend logging in to [Confident AI](https://app.confident-ai.com) to view your LlamaIndex evaluation traces.
"""
logger.info("# LlamaIndex")

deepeval login

"""
:::

## End-to-End Evals

`deepeval` allows you to evaluate LlamaIndex applications end-to-end in **under a minute**.

<Timeline>

<TimelineItem title="Configure LlamaIndex">

Create a `FunctionAgent` with a list of metrics you wish to use, and pass it to your LlamaIndex application's `run` method.
"""
logger.info("## End-to-End Evals")




answer_relevance_metric = AnswerRelevancyMetric()

instrument_llama_index(instrument.get_dispatcher())

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(model="llama3.2"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevance_metric]
)

async def llm_app(input: str):
    return await agent.run(input)

"""
:::info
Evaluations are supported for LlamaIndex `FunctionAgent`, `ReActAgent` and `CodeActAgent`. Only metrics with LLM parameters `input` and `output` are eligible for evaluation.
:::

</TimelineItem>
<TimelineItem title="Run evaluations">

Create an `EvaluationDataset` and invoke your LlamaIndex application for each golden within the `evals_iterator()` loop to run end-to-end evaluations.

<Tabs groupId="llamaindex">
<TabItem value="asynchronous" label="Asynchronous">
"""
logger.info("Evaluations are supported for LlamaIndex `FunctionAgent`, `ReActAgent` and `CodeActAgent`. Only metrics with LLM parameters `input` and `output` are eligible for evaluation.")


dataset = EvaluationDataset(goldens=[
    Golden(input="What is 3 * 12?"),
    Golden(input="What is 4 * 13?")
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input))
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

✅ Done. The `evals_iterator` will automatically generate a test run with individual evaluation traces for each golden.

</TimelineItem>
<TimelineItem title="View on Confident AI (optional)">

<VideoDisplayer
  src="https://confident-bucket.s3.us-east-1.amazonaws.com/end-to-end%3Allama-index-1080.mp4"
/>

</TimelineItem>

</Timeline>

:::note
If you need to evaluate individual components of your LlamaIndex application, [set up tracing](/docs/evaluation-llm-tracing) instead.
:::

## Evals in Production

To run online evaluations in production, simply replace `metrics` in `FunctionAgent` with a [metric collection](https://documentation.confident-ai.com/docs/llm-tracing/evaluations#online-evaluations) string from Confident AI, and push your LlamaIndex agent to production.
"""
logger.info("## Evals in Production")

...

agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(model="llama3.2"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1"
)

agent.run("What is 3 * 12?")

logger.info("\n\n[DONE]", bright=True)