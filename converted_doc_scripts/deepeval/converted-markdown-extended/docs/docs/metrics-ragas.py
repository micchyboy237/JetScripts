from deepeval import evaluate
from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric
from deepeval.metrics.ragas import RAGASContextualPrecisionMetric
from deepeval.metrics.ragas import RAGASContextualRecallMetric
from deepeval.metrics.ragas import RAGASFaithfulnessMetric
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
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
id: metrics-ragas
title: RAGAS
sidebar_label: RAGAS
---

<head>
  <link rel="canonical" href="https://deepeval.com/docs/metrics-ragas" />
</head>

The RAGAS metric is the average of four distinct metrics:

- `RAGASAnswerRelevancyMetric`
- `RAGASFaithfulnessMetric`
- `RAGASContextualPrecisionMetric`
- `RAGASContextualRecallMetric`

It provides a score to holistically evaluate of your RAG pipeline's generator and retriever.

:::info WHAT'S THE DIFFERENCE?
The `RAGASMetric` uses the `ragas` library under the hood and are available on `deepeval` with the intention to allow users of `deepeval` can have access to `ragas` in `deepeval`'s ecosystem as well. They are implemented in an almost identical way to `deepeval`'s default RAG metrics. However there are a few differences, including but not limited to:

- `deepeval`'s RAG metrics generates a reason that corresponds to the score equation. Although both `ragas` and `deepeval` has equations attached to their default metrics, `deepeval` incorporates an LLM judges' reasoning along the way.
- `deepeval`'s RAG metrics are debuggable - meaning you can inspect the LLM judges' judgements along the way to see why the score is a certain way.
- `deepeval`'s RAG metrics are JSON confineable. You'll often meet `NaN` scores in `ragas` because of invalid JSONs generated - but `deepeval` offers a way for you to use literally any custom LLM for evaluation and [JSON confine them in a few lines of code.](/guides/guides-using-custom-llms)
- `deepeval`'s RAG metrics integrates **fully** with `deepeval`'s ecosystem. This means you'll get access to metrics caching, native support for `pytest` integrations, first-class error handling, available on Confident AI, and so much more.

Due to these reasons, we highly recommend that you use `deepeval`'s RAG metrics instead. They're proven to work, and if not better according to [examples shown in some studies.](https://arxiv.org/pdf/2409.06595)

:::

## Required Arguments

To use the `RagasMetric`, you'll have to provide the following arguments when creating an [`LLMTestCase`](/docs/evaluation-test-cases#llm-test-case):

- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`

## Usage

First, install `ragas`:
"""
logger.info("## Required Arguments")

pip install ragas

"""
Then, use it within `deepeval`:
"""
logger.info("Then, use it within `deepeval`:")


actual_output = "We offer a 30-day full refund at no extra cost."

expected_output = "You are eligible for a 30 day full refund at no extra cost."

retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = RagasMetric(threshold=0.5, model="llama3.2")
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
logger.debug(metric.score)

evaluate([test_case], [metric])

"""
There are **THREE** optional parameters when creating a `RagasMetric`:

- [Optional] `threshold`: a float representing the minimum passing threshold, defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** any one of langchain's [chat models](https://python.langchain.com/docs/integrations/chat/) of type `BaseChatModel`. Defaulted to 'gpt-3.5-turbo'.
- [Optional] `embeddings`: any one of langchain's [embedding models](https://python.langchain.com/docs/integrations/text_embedding) of type `Embeddings`. Custom `embeddings` provided to the `RagasMetric` will only be used in the `RAGASAnswerRelevancyMetric`, since it is the only metric that requires embeddings for calculating cosine similarity.

:::info
You can also choose to import and execute each metric individually:
"""
logger.info("There are **THREE** optional parameters when creating a `RagasMetric`:")


"""
These metrics accept the same arguments as the `RagasMetric`.
:::
"""
logger.info("These metrics accept the same arguments as the `RagasMetric`.")

logger.info("\n\n[DONE]", bright=True)