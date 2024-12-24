from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/Deepeval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 🚀 RAG/LLM Evaluators - DeepEval
# 
# This code tutorial shows how you can easily integrate DeepEval with LlamaIndex. DeepEval makes it easy to unit-test your RAG/LLMs.
# 
# You can read more about the DeepEval framework here: https://docs.confident-ai.com/docs/getting-started
# 
# Feel free to check out our repository here on GitHub: https://github.com/confident-ai/deepeval

### Set-up and Installation
# 
# We recommend setting up and installing via pip!

# !pip install -q -q llama-index
# !pip install -U -q deepeval

# This step is optional and only if you want a server-hosted dashboard! (Psst I think you should!)

# !deepeval login

## Types of Metrics
# 
# DeepEval presents an opinionated framework for unit testing RAG applications. It breaks down evaluations into test cases, and offers a range of evaluation metrics that you can freely evaluate for each test case, including:
# 
# - G-Eval
# - Summarization
# - Answer Relevancy
# - Faithfulness
# - Contextual Recall
# - Contextual Precision
# - Contextual Relevancy
# - RAGAS
# - Hallucination
# - Bias
# - Toxicity
# 
# [DeepEval](https://github.com/confident-ai/deepeval) incorporates the latest research into its evaluation metrics, which are then used to power LlamaIndex's evaluators. You can learn more about the full list of metrics and how they are calculated [here.](https://docs.confident-ai.com/docs/metrics-introduction)

## Step 1 - Setting Up Your LlamaIndex Application

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
rag_application = index.as_query_engine()

## Step 2 - Using DeepEval's RAG/LLM evaluators

# DeepEval offers 6 evaluators out of the box, some for RAG, some directly for LLM outputs (although also works for RAG). Let's try the faithfulness evaluator (which is for evaluating hallucination in RAG):

from deepeval.integrations.llamaindex import DeepEvalFaithfulnessEvaluator

user_input = "What is LlamaIndex?"

response_object = rag_application.query(user_input)

evaluator = DeepEvalFaithfulnessEvaluator()
evaluation_result = evaluator.evaluate_response(
    query=user_input, response=response_object
)
print(evaluation_result)

## Full List of Evaluators
# 
# Here is how you can import all 6 evaluators from `deepeval`:
# 
# ```python
# from deepeval.integrations.llama_index import (
#     DeepEvalAnswerRelevancyEvaluator,
#     DeepEvalFaithfulnessEvaluator,
#     DeepEvalContextualRelevancyEvaluator,
#     DeepEvalSummarizationEvaluator,
#     DeepEvalBiasEvaluator,
#     DeepEvalToxicityEvaluator,
# )
# ```
# 
# For all evaluator definitions and to understand how it integrates with DeepEval's testing suite, [click here.](https://docs.confident-ai.com/docs/integrations-llamaindex)
# 
## Useful Links
# 
# - [DeepEval Quickstart](https://docs.confident-ai.com/docs/getting-started)
# - [Everything you need to know about LLM evaluation metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

logger.info("\n\n[DONE]", bright=True)