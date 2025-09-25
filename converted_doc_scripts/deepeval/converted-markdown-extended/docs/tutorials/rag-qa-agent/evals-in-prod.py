from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
ContextualRelevancyMetric,
ContextualRecallMetric,
ContextualPrecisionMetric,
GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.tracing import observe, update_current_span
from jet.logger import logger
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from qa_agent import RAGAgent # import your RAG agent here
import os
import pytest
import shutil
import tempfile


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
id: evals-in-prod
title: Deployment
sidebar_label: Deploy And Run Evals in Prod
---

In this section we'll set up CI/CD workflows for our RAG QA agent. We'll also see how to add metrics and create spans in our RAG agent's `@observe` decorators to do online evals and get full visibilty for debugging internal components.

## Setup Tracing

`deepeval` offers an `@observe` decorator for you to apply metrics at any point in your LLM app to evaluate any [LLM interaction](https://deepeval.com/docs/evaluation-test-cases#what-is-an-llm-interaction),
this provides full visibility for debugging internal components of your LLM application. [Learn more about tracing here](https://deepeval.com/docs/evaluation-llm-tracing).

During our development phase, we've added these `@observe` decorators to our RAG agent for different components, we will now add metrics and create spans. Here's how you can do that:
"""
logger.info("## Setup Tracing")


class RAGAgent:
    def __init__(...):
        ...

    def _load_vector_store(self):
        ...

    @observe(metrics=[ContextualRelevancyMetric(), ContextualRecallMetric(), ContextualPrecisionMetric()], name="Retriever")
    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output="...",
                expected_output="...",
                retrieval_context=context
            )
        )
        return context

    @observe(metrics=[GEval(...), GEval(...)], name="Generator") # Use same metrics as before
    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None
    ): # Changed prompt template, model used
        context = "\n".join(retrieved_docs)
        model = llm_model or Ollama(model_name="gpt-4")
        prompt = prompt_template or (
            "You are an AI assistant designed for factual retrieval. Using the context below, extract only the information needed to answer the user's query. Respond in strictly valid JSON using the schema below.\n\nResponse schema:\n{\n  \"answer\": \"string — a precise, factual answer found in the context\",\n  \"citations\": [\n    \"string — exact quotes or summaries from the context that support the answer\"\n  ]\n}\n\nRules:\n- Do not fabricate any information or cite anything not present in the context.\n- Do not include explanations or formatting — only return valid JSON.\n- Use complete sentences in the answer.\n- Limit the answer to the scope of the context.\n- If no answer is found in the context, return:\n{\n  \"answer\": \"No relevant information available.\",\n  \"citations\": []\n}\n\nContext:\n{context}\n\nQuery:\n{query}"
        )
        prompt = prompt.format(context=context, query=query)
        answer = model(prompt)
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieved_docs
            )
        )
        return answer

    @observe(type="agent")
    def answer(
        self,
        query: str,
        llm_model=None,
        prompt_template: str = None
    ):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs, llm_model, prompt_template)
        return generated_answer, retrieved_docs

"""
## Using Datasets

In the previous section, we've seen how to create datasets and store them in the cloud. We can now pull that dataset and use it in the CI/CD to evaluate our RAG agent.

Here's how we can pull datasets from the cloud:
"""
logger.info("## Using Datasets")


dataset = EvaluationDataset()
dataset.pull(alias="QA Agent Dataset")

"""
## Integrating CI/CD

You can use `pytest` with `assert_test` during your CI/CD to trace and evaluate your RAG agent, here's how you can write the test file to do that:
"""
logger.info("## Integrating CI/CD")


dataset = EvaluationDataset()
dataset.pull(alias="QA Agent Dataset")

agent = RAGAgent() # Initialize with your best config

@pytest.mark.parametrize("golden", dataset.goldens)
def test_meeting_summarizer_components(golden):
    assert_test(golden=golden, observed_callback=agent.answer)

"""

"""

poetry run deepeval test run test_rag_qa_agent.py

"""
Finally, let's integrate this test into GitHub Actions to enable automated quality checks on every push.
"""
logger.info("Finally, let's integrate this test into GitHub Actions to enable automated quality checks on every push.")

name: RAG QA Agent DeepEval Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install Dependencies
        run: poetry install --no-root

      - name: Run DeepEval Unit Tests
        env:
#           OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} # Add your OPENAI_API_KEY
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }} # Add your CONFIDENT_API_KEY
        run: poetry run deepeval test run test_rag_qa_agent.py

"""
And that's it! You now have a reliable, production-ready RAG QA agent with automated evaluation integrated into your development workflow.

:::tip Next Steps
Setup [Confident AI](https://deepeval.com/tutorials/tutorial-setup) to track your RAG QA agent's performance across builds, regressions, and evolving datasets. **It's free to get started.** _(No credit card required)_

Learn more [here](https://www.confident-ai.com).
:::
"""
logger.info("And that's it! You now have a reliable, production-ready RAG QA agent with automated evaluation integrated into your development workflow.")

logger.info("\n\n[DONE]", bright=True)