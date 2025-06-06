from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/evaluation/evaluation_deep_eval.ipynb)

# Deep Evaluation of RAG Systems using deepeval

## Overview

This code demonstrates the use of the `deepeval` library to perform comprehensive evaluations of Retrieval-Augmented Generation (RAG) systems. It covers various evaluation metrics and provides a framework for creating and running test cases.

## Key Components

1. Correctness Evaluation
2. Faithfulness Evaluation
3. Contextual Relevancy Evaluation
4. Combined Evaluation of Multiple Metrics
5. Batch Test Case Creation

## Evaluation Metrics

### 1. Correctness (GEval)

- Evaluates whether the actual output is factually correct based on the expected output.
- Uses GPT-4 as the evaluation model.
- Compares the expected and actual outputs.

### 2. Faithfulness (FaithfulnessMetric)

- Assesses whether the generated answer is faithful to the provided context.
- Uses GPT-4 as the evaluation model.
- Can provide detailed reasons for the evaluation.

### 3. Contextual Relevancy (ContextualRelevancyMetric)

- Evaluates how relevant the retrieved context is to the question and answer.
- Uses GPT-4 as the evaluation model.
- Can provide detailed reasons for the evaluation.

## Key Features

1. Flexible Metric Configuration: Each metric can be customized with different models and parameters.
2. Multi-Metric Evaluation: Ability to evaluate test cases using multiple metrics simultaneously.
3. Batch Test Case Creation: Utility function to create multiple test cases efficiently.
4. Detailed Feedback: Options to include detailed reasons for evaluation results.

## Benefits of this Approach

1. Comprehensive Evaluation: Covers multiple aspects of RAG system performance.
2. Flexibility: Easy to add or modify evaluation metrics and test cases.
3. Scalability: Capable of handling multiple test cases and metrics efficiently.
4. Interpretability: Provides detailed reasons for evaluation results, aiding in system improvement.

## Conclusion

This deep evaluation approach using the `deepeval` library offers a robust framework for assessing the performance of RAG systems. By evaluating correctness, faithfulness, and contextual relevancy, it provides a multi-faceted view of system performance. This comprehensive evaluation is crucial for identifying areas of improvement and ensuring the reliability and effectiveness of RAG systems in real-world applications.
"""
logger.info("# Deep Evaluation of RAG Systems using deepeval")


"""
### Test Correctness
"""
logger.info("### Test Correctness")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.1", request_timeout=300.0, context_window=4096,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],

)

gt_answer = "Madrid is the capital of Spain."
pred_answer = "MadriD."

test_case_correctness = LLMTestCase(
    input="What is the capital of Spain?",
    expected_output=gt_answer,
    actual_output=pred_answer,
)

correctness_metric.measure(test_case_correctness)
logger.debug(correctness_metric.score)

"""
### Test faithfulness
"""
logger.info("### Test faithfulness")

question = "what is 3+3?"
context = ["6"]
generated_answer = "6"

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="llama3.1", request_timeout=300.0, context_window=4096,
    include_reason=False
)

test_case = LLMTestCase(
    input = question,
    actual_output=generated_answer,
    retrieval_context=context

)

faithfulness_metric.measure(test_case)
logger.debug(faithfulness_metric.score)
logger.debug(faithfulness_metric.reason)

"""
### Test contextual relevancy
"""
logger.info("### Test contextual relevancy")

actual_output = "then go somewhere else."
retrieval_context = ["this is a test context","mike is a cat","if the shoes don't fit, then go somewhere else."]
gt_answer = "if the shoes don't fit, then go somewhere else."

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="llama3.1", request_timeout=300.0, context_window=4096,
    include_reason=True
)
relevance_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output=gt_answer,

)

relevance_metric.measure(relevance_test_case)
logger.debug(relevance_metric.score)
logger.debug(relevance_metric.reason)

new_test_case = LLMTestCase(
    input="What is the capital of Spain?",
    expected_output="Madrid is the capital of Spain.",
    actual_output="MadriD.",
    retrieval_context=["Madrid is the capital of Spain."]
)

"""
### Test two different cases together with several metrics together
"""
logger.info("### Test two different cases together with several metrics together")

evaluate(
    test_cases=[relevance_test_case, new_test_case],
    metrics=[correctness_metric, faithfulness_metric, relevance_metric]
)

"""
### Funcion to create multiple LLMTestCases based on four lists: 
* Questions
* Ground Truth Answers
* Generated Answers
* Retrieved Documents - Each element is a list
"""
logger.info("### Funcion to create multiple LLMTestCases based on four lists:")

def create_deep_eval_test_cases(questions, gt_answers, generated_answers, retrieved_documents):
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

logger.info("\n\n[DONE]", bright=True)