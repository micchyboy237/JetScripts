from jet.transformers.formatters import format_json
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.ollama import AsyncOpenAI
from deepeval.ollama import Ollama
from deepeval.tracing import observe
from jet.logger import logger
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import asyncio
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
id: ollama
title: Ollama
sidebar_label: Ollama
---


`deepeval` streamlines the process of evaluating and tracing your Ollama applications through an **Ollama client wrapper**, and supports both end-to-end and component-level evaluations, and online evaluations in production.

## End-to-End Evals

To begin evaluating your Ollama application, simply replace your Ollama client with `deepeval`'s Ollama client, and pass in the `metrics` you wish to use.

<Tabs groupId="ollama">
<TabItem value="chat-completions" label="Chat Completions">
"""
logger.info("## End-to-End Evals")


dataset = EvaluationDataset(goldens=[Golden(input="Test")])

client = Ollama()

for golden in dataset.evals_iterator():
    client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": golden.input}
        ],
        metrics=[AnswerRelevancyMetric(), BiasMetric()]
    )

"""
</TabItem>
<TabItem value="responses" label="Responses">
"""


dataset = EvaluationDataset(goldens=[Golden(input="Test")])

client = Ollama()

for golden in dataset.evals_iterator():
    client.responses.create(
        model="llama3.2",
        instructions="You are a helpful assistant.",
        input=golden.input,
        metrics=[AnswerRelevancyMetric(), BiasMetric()]
    )

"""
</TabItem>
<TabItem value="async-chat-completions" label="Async Chat Completions">
"""


dataset = EvaluationDataset(goldens=[Golden(input="Test")])

client = AsyncOpenAI()

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": golden.input}
            ],
            metrics=[AnswerRelevancyMetric(), BiasMetric()]
        )
    )
    dataset.evaluate(task)

"""
</TabItem>
<TabItem value="async-responses" label="Async Responses">
"""


dataset = EvaluationDataset(goldens=[Golden(input="Test")])

client = AsyncOpenAI()

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        client.responses.create(
            model="llama3.2",
            instructions="You are a helpful assistant.",
            input=golden.input,
            metrics=[AnswerRelevancyMetric(), BiasMetric()]
        )
    )
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

There are **FIVE** optional parameters when using `deepeval`'s Ollama client's chat completion and response methods:

- [Optional] `metrics`: a list of metrics of type `BaseMetric`
- [Optional] `expected_output`: a string specifying the expected output of your Ollama generation.
- [Optional] `retrieval_context`: a list of strings, representing the retrieved contexts to be passed into your Ollama generation.
- [Optional] `context`: a list of strings, representing the ideal retrieved contexts to be passed into your Ollama generation.
- [Optional] `expected_tools`: a list of strings, representing the expected tools to be called during Ollama generation.

:::info
`deepeval` Ollama client automatically extracts the `input` and `actual_output` from each API response, enabling you to use metrics like **Answer Relevancy** out of the box. For metrics such as **Faithfulness**—which rely on additional parameters such as retrieval context—you’ll need to explicitly set these parameters when invoking the client.
:::

## Component-Level Evals

You can also use `deepeval`'s Ollama client **within component-level evaluations**. To set up component-level evaluations, add the `@observe` decorator to your llm_application's components, and simply replace existing Ollama clients with `deepeval`'s Ollama client, passing in the metrics you wish to use.

<Tabs groupId="ollama">
<TabItem value="chat-completions" label="Chat Completions">
"""
logger.info("## Component-Level Evals")


client = Ollama()

@observe()
def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
def llm_app(input):
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '\n'.join(retrieve_docs(input)) + "\n\nQuestion: " + input}
        ],
        metrics=[AnswerRelevancyMetric(), BiasMetric()]
    )
    return response.choices[0].message.content

dataset = EvaluationDataset(goldens=[Golden(input="...")])

for golden in dataset.evals_iterator():
    llm_app(input=golden.input)

"""
</TabItem>
<TabItem value="responses" label="Responses">
"""


@observe()
def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
def llm_app(input):
    client = Ollama()
    response = client.responses.create(
        model="llama3.2",
        instructions="You are a helpful assistant.",
        input=input,
        metrics=[AnswerRelevancyMetric(), BiasMetric()]
    )
    return response.output_text

dataset = EvaluationDataset(goldens=[Golden(input="...")])

for golden in dataset.evals_iterator():
    llm_app(input=golden.input)

"""
</TabItem>
<TabItem value="async-chat-completions" label="Async Chat Completions">
"""


@observe()
async def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
async def llm_app(input):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": '\n'.join(await retrieve_docs(input)) + "\n\nQuestion: " + input}
            ],
            metrics=[AnswerRelevancyMetric(), BiasMetric()]
        )
    logger.success(format_json(response))
    return response.choices[0].message.content

dataset = EvaluationDataset(goldens=[Golden(input="...")])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)

"""
</TabItem>
<TabItem value="async-responses" label="Async Responses">
"""


@observe()
async def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
async def llm_app(input):
    client = AsyncOpenAI()
    response = await client.responses.create(
            model="llama3.2",
            instructions="You are a helpful assistant.",
            input=input,
            metrics=[AnswerRelevancyMetric(), BiasMetric()]
        )
    logger.success(format_json(response))
    return response.output_text

dataset = EvaluationDataset(goldens=[Golden(input="...")])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)

"""
</TabItem>
</Tabs>

When used inside `@observe` components, `deepeval`'s Ollama client automatically:

- Generates an LLM span for every Ollama API call, including nested Tool spans for any tool invocations.
- Attaches an `LLMTestCase` to each generated LLM span, capturing inputs, outputs, and tools called.
- Records span-level llm attributes such as the input prompt, generated output and token usage.
- Logs hyperparameters such as model name and system prompt for comprehensive experiment analysis.

<div style={{ margin: "2rem 0" }}>
  <VideoDisplayer
    src="https://deepeval-docs.s3.us-east-1.amazonaws.com/integrations:frameworks:ollama.mp4"
    label="Ollama Integration"
    confidentUrl="/llm-tracing/integrations/ollama"
  />
</div>

## Online Evals in Production

...To be documented
"""
logger.info("## Online Evals in Production")

logger.info("\n\n[DONE]", bright=True)