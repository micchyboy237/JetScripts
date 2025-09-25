from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe
from deepeval.tracing import observe, update_current_span
from deepeval.tracing import observe, update_current_trace
from jet.logger import logger
from ollama import Ollama
import NavigationCards from "@site/src/components/NavigationCards";
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
id: evaluation-llm-tracing
title: LLM Tracing
sidebar_label: Tracing
---


Tracing your LLM application helps you monitor its full execution from start to finish. With `deepeval`'s `@observe` decorator, you can trace and evaluate any [LLM interaction](/docs/evaluation-test-cases#what-is-an-llm-interaction) at any point in your app no matter how complex they may be.

## Quick Summary

An LLM trace is made up of multiple individual spans. A **span** is a flexible, user-defined scope for evaluation or debugging. A full **trace** of your application contains one or more spans.

![LLM Trace](https://deepeval-docs.s3.amazonaws.com/docs:llm-trace.png)

Tracing allows you run both [end-to-end](https://www.deepeval.com/docs/evaluation-end-to-end-llm-evals) and [component level](https://www.deepeval.com/docs/evaluation-component-level-llm-evals) evals which you'll learn about in the later sections.

<details>

<summary>Learn how DeepEval's tracing is non-instrusive</summary>

`deepeval`'s tracing is **non-intrusive**, it requires **minimal code change** and **doesn't add latency** to your LLM application. It also:

- **Uses concepts you already know**: Tracing a component in your LLM app takes on average 3 lines of code, which uses the same `LLMTestCase`s and [metrics](/docs/metrics-introduction) that you're already familiar with.

- **Does not affect production code**: If you're worried that tracing will affect your LLM calls in production, it won't. This is because the `@observe` decorators that you add for tracing is only invoked if called explicitly during evaluation.

- **Non-opinionated**: `deepeval` does not care what you consider a "component" - in fact a component can be anything, at any scope, as long as you're able to set your `LLMTestCase` within that scope for evaluation.

Tracing only runs when you want it to run, and takes 3 lines of code:
"""
logger.info("## Quick Summary")


client = Ollama()

@observe(metrics=[AnswerRelevancyMetric()])
def get_res(query: str):
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content

    update_current_span(input=query, output=response)
    return response

"""
</details>

## Why Tracing?

Tracing your LLM applications allows you to:

- **Generate test cases dynamically:** Many components rely on upstream outputs. Tracing lets you define `LLMTestCase`s at runtime as data flows through the system.

- **Debug with precision:** See exactly where and why things fail—whether it’s tool calls, intermediate outputs, or context retrieval steps.

- **Run targeted metrics on specific components:** Attach `LLMTestCase`s to agents, tools, retrievers, or LLMs and apply metrics like answer relevancy or context precision—without needing to restructure your app.

- **Run end-to-end evals with trace data:** Use the `evals_iterator` with `metrics` to perform comprehensive evaluations using your traces.

## Setup Your First Trace

To set up tracing in your LLM app, you need to understand two key concepts:

- **Trace**: The full execution of your app, made up of one or more spans.
- **Span**: A specific component or unit of work—like an LLM call, tool invocation, or document retrieval.

The [`@observe`](#observe) decorator is the primary way to setup tracing for your LLM application.

<Timeline>
<TimelineItem title="Decorate your components">

An individual function that makes up a part of your LLM application or is invoked only when necessary, can be classified as a **component**. You can decorate this component with `deepeval`'s `@observe` decorator.
"""
logger.info("## Why Tracing?")


client = Ollama()

@observe()
def get_res(query: str):
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content

    return response

"""
The above `get_res()` component is treated as an individual `span` within a `trace`.

</TimelineItem>
<TimelineItem title="Add test cases inside components">

You can assign individual test cases to a `span` by using the [`update_current_span`](#update-current-span) function from `deepeval`. This allows you to create separate `LLMTestCase`s on a component level.
"""
logger.info("The above `get_res()` component is treated as an individual `span` within a `trace`.")


client = Ollama()

@observe()
def get_res(query: str):
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content

    update_current_span(input=query, output=response)
    return response

"""
You can either supply the `LLMTestCase` or its parameters in the `update_current_span` to create a component level test case. Learn more [here](#update-current-span).

</TimelineItem>
<TimelineItem title="Get your traces">

You can now get your traces by simply calling your observed function or application.
"""
logger.info("You can either supply the `LLMTestCase` or its parameters in the `update_current_span` to create a component level test case. Learn more [here](#update-current-span).")

query = "This will get you a trace."

get_res(query)

"""
🎉🥳 **Congratulations!** You just created your first trace with `deepeval`.

:::tip
We highly recommend setting up Confident AI to look at your traces in an intuitive UI like this:

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/llm-tracing:traces.mp4"
  confidentUrl="/docs/llm-tracing/introduction"
  label="Learn how to setup LLM tracing for Confident AI"
/>

It's free to get started. Just the following command:
"""
logger.info("We highly recommend setting up Confident AI to look at your traces in an intuitive UI like this:")

deepeval login

"""
:::

</TimelineItem>
</Timeline>

### Observe

The `@observe` decorator is a non-intrusive python decorator that you can use on top of any component as you wish. It tracks the usage of the component whenever it is evoked to create a span.

A span can contain many child spans, forming a tree structure—just like how different components of your LLM application interact
"""
logger.info("### Observe")


@observe()
def generate(query: str) -> str:
    context = retrieve(query)
    return f"Output for given {query} and {context}."

@observe()
def retrieve(query: str) -> str:
    return [f"Context for the given {query}"]

"""
From the above example, an observed component `generate` calling another observed component `retrieve` create a nested span `generate` with `retrieve` inside it.

There are **FOUR** optional parameters when using the `@observe` decorator:

- [Optional] `metrics`: A list of metrics of type `BaseMetric` that will be used to evaluate your span.
- [Optional] `name`: The function name or a string specifying how this span is displayed on Confident AI.
- [Optional] `type`: A string specifying the type of span. The value can be any one of `llm`, `retriever`, `tool`, and `agent`. Any other value is treated as a custom span type.
- [Optional] `metric_collection`: The name of the metric collection you stored on Confident AI.

<details>
<summary><strong>Click here to learn more about span types</strong></summary>

For simplicity, we always recommend **custom spans** unless needed otherwise, since `metrics` only care about the scope of the span, and supplying a specified `type` is most **useful only when using Confident AI**. To summarize:

- Specifying a span type (like `"llm"`) allows you to supply additional parameters in the `@observe` signature (e.g., the `model` used).
- This information becomes extremely useful for analysis and visualization if you're using `deepeval` together with **Confident AI** (highly recommended).
- Otherwise, for local evaluation purposes, span `type` makes **no difference** — evaluation still works the same way.

To learn more about the different spans `type`s, or to run LLM evaluations with tracing with an UI for visualization and debugging, visiting the [official Confident AI docs on LLM tracing.](https://www.confident-ai.com/docs/llm-tracing/introduction)

</details>

### Update Current Span

The `update_current_span` method can be used to create a test case for the corresponding span. This is especially useful for doing component level evals or debugging your application.
"""
logger.info("### Update Current Span")


@observe()
def generate(query: str) -> str:
    context = retrieve(query)
    res = f"Output for given {query} and {context}."
    update_current_span(test_case=LLMTestCase(
        input=query,
        actual_output=res,
        retrieval_context=context
    ))
    return res

@observe()
def retrieve(query: str) -> str:
    context = [f"Context for the given {query}"]
    update_current_span(input=query, retrieval_context=context)
    return context

"""
There are **TWO** ways to create test cases when using the `update_current_span` function:

- [Optional] `test_case`: Takes an `LLMTestCase` to create a span level test case for that component.

- Or, You can also opt to give the values of `LLMTestCase` directly by using the following attributes:
  - [Optional] `input`
  - [Optional] `output`
  - [Optional] `retrieval_context`
  - [Optional] `context`
  - [Optional] `expected_output`
  - [Optional] `tools_called`
  - [Optional] `expected_tools`

:::note
You can use the individual `LLMTestCase` params in the `update_current_span` function to override the values of the `test_case` you passed.
:::

### Update Current Trace

You can update your end-to-end test cases for trace by using the `update_current_trace` function provided by `deepeval`
"""
logger.info("### Update Current Trace")


@observe()
def llm_app(query: str) -> str:

    @observe()
    def retriever(query: str) -> list[str]:
        chunks = ["List", "of", "text", "chunks"]
        update_current_trace(retrieval_context=chunks)
        return chunks

    @observe()
    def generator(query: str, text_chunks: list[str]) -> str:
        res = Ollama().chat.completions.create(model="llama3.2", messages=[{"role": "user", "content": query}]
        ).choices[0].message.content
        update_current_trace(input=query, output=res)
        return res

    return generator(query, retriever(query))

"""
There are **TWO** ways to create test cases when using the `update_current_trace` function:

- [Optional] `test_case`: Takes an `LLMTestCase` to create a span level test case for that component.

- Or, You can also opt to give the values of `LLMTestCase` directly by using the following attributes:
  - [Optional] `input`
  - [Optional] `output`
  - [Optional] `retrieval_context`
  - [Optional] `context`
  - [Optional] `expected_output`
  - [Optional] `tools_called`
  - [Optional] `expected_tools`

:::note
You can use the individual `LLMTestCase` params in the `update_current_trace` function to override the values of the `test_case` you passed.
:::

## Environment Variables

If you run your `@observe` decorated LLM application outside of `evaluate()` or `assert_test()`, you'll notice some logs appearing in your console. To disable them completely, just set the following environment variables:
"""
logger.info("## Environment Variables")

CONFIDENT_TRACE_VERBOSE=0
CONFIDENT_TRACE_FLUSH=0

"""
## Next Steps

Now that you have your traces, you can run either end-to-end or component level evals.

<NavigationCards
  items={[
    {
      title: "End-to-End Evals",
      description: "Learn how to run end-to-end evals with your trace data,",
      icon: "SendToBack",
      to: "/docs/evaluation-end-to-end-llm-evals#use-evaluate-in-python-scripts",
    },
    {
      title: "Component-Level Evals",
      description: "Learn how to run component-level evals using tracing.",
      icon: "ArrowDownWideNarrow",
      to: "/docs/evaluation-component-level-llm-evals#use-python-scripts",
    },
  ]}
/>
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)