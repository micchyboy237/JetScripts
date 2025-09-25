from deepeval.dataset import Golden, EvaluationDataset
from deepeval.integrations.langchain import CallbackHandler
from deepeval.integrations.langchain import tool
from deepeval.metrics import TaskCompletionMetric
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langgraph.prebuilt import create_react_agent
import ColabButton from "@site/src/components/ColabButton";
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import asyncio
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
id: langgraph
title: LangGraph
sidebar_label: LangGraph
---


# LangGraph

<ColabButton 
  notebookUrl="https://colab.research.google.com/github/confident-ai/deepeval/blob/main/cookbooks/examples/notebooks/langgraph.ipynb" 
  className="header-colab-button"
/>

[LangGraph](https://www.langchain.com/langgraph) is an open-source framework for developing applications powered by large language models, enabling chaining of LLMs with external data sources and expressive workflows to build advanced generative AI solutions.

:::tip
We recommend logging in to [Confident AI](https://app.confident-ai.com) to view your LangGraph evaluation traces.
"""
logger.info("# LangGraph")

deepeval login

"""
:::

## End-to-End Evals

`deepeval` allows you to evaluate LangGraph applications end-to-end in **under a minute**.

<Timeline>

<TimelineItem title="Configure LangGraph">

Create a `CallbackHandler` with a list of [task completion metrics](/docs/metrics-task-completion) you wish to use, and pass it to your LangGraph application's `invoke` method.
"""
logger.info("## End-to-End Evals")



task_completion_metric = TaskCompletionMetric()

def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="ollama:llama3.2",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

"""
:::info
Only [Task Completion](/docs/metrics-task-completion) is supported for the LangGraph integration. To use other metrics, manually [set up tracing](/docs/evaluation-llm-tracing) instead.
:::

</TimelineItem>
<TimelineItem title="Run evaluations">

Create an `EvaluationDataset` and invoke your LangGraph application for each golden within the `evals_iterator()` loop to run end-to-end evaluations.

<Tabs groupId="langgraph">
<TabItem value="synchronous" label="Synchronous">
"""
logger.info("Only [Task Completion](/docs/metrics-task-completion) is supported for the LangGraph integration. To use other metrics, manually [set up tracing](/docs/evaluation-llm-tracing) instead.")


goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]

dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler(metrics=[task_completion_metric])]}
    )

"""
</TabItem>
  <TabItem value="asynchronous" label="Asynchronous">
"""


dataset = EvaluationDataset(goldens=[
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent.ainvoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={"callbacks": [CallbackHandler(metrics=[task_completion_metric])]}
        )
    )
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

✅ Done. The `evals_iterator` will automatically generate a test run with individual evaluation traces for each golden.

</TimelineItem>
<TimelineItem title="View on Confident AI (optional)">

<VideoDisplayer
  src="https://confident-bucket.s3.us-east-1.amazonaws.com/end-to-end%3Alanggraph.mp4"
/>

</TimelineItem>

</Timeline>

:::note
If you need to evaluate individual components of your LangGraph application, [set up tracing](/docs/evaluation-llm-tracing) instead.
:::

## Component-level Evals

Using `deepeval`, you can now evaluate individual components of your LangGraph application.

### LLM
Define `metrics` or `metric_collection` in the metadata of the all the `BaseLanguageModel`s in your LangGraph application.
"""
logger.info("## Component-level Evals")

...

llm = ChatOllama(
    model="llama3.2",
    metadata={"metric_collection": "test_collection_1"}
).bind_tools([get_weather])

"""
### Tool
To pass `metrics` or `metric_collection` to the tools, you can use the DeepEval's LangChain `tool` decorator.
"""
logger.info("### Tool")

...

@tool(metric_collection="test_collection_1")
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"It's always sunny in {location}!"

"""
## Evals in Production

To run online evaluations in production, simply replace `
"""
logger.info("## Evals in Production")

logger.info("\n\n[DONE]", bright=True)