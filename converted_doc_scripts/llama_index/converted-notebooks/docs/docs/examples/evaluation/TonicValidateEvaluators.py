from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.evaluation.tonic_validate import (
    AnswerConsistencyEvaluator,
    AnswerSimilarityEvaluator,
    AugmentationAccuracyEvaluator,
    AugmentationPrecisionEvaluator,
    RetrievalPrecisionEvaluator,
    TonicValidateEvaluator,
)
from tonic_validate.metrics import AnswerSimilarityMetric
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a target="_blank" href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/TonicValidateEvaluators.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Tonic Validate Evaluators

This notebook has some basic usage examples of how to use [Tonic Validate](https://github.com/TonicAI/tonic_validate)'s RAGs metrics using LlamaIndex. To use these evaluators, you need to have `tonic_validate` installed, which you can install via `pip install tonic-validate`.
"""
logger.info("# Tonic Validate Evaluators")

# %pip install llama-index-evaluation-tonic-validate


"""
## One Question Usage Example

For this example, we have an example of a question with a reference correct answer that does not match the LLM response answer. There are two retrieved context chunks, of which one of them has the correct answer.
"""
logger.info("## One Question Usage Example")

question = "What makes Sam Altman a good founder?"
reference_answer = "He is smart and has a great force of will."
llm_answer = "He is a good founder because he is smart."
retrieved_context_list = [
    "Sam Altman is a good founder. He is very smart.",
    "What makes Sam Altman such a good founder is his great force of will.",
]

"""
The answer similarity score is a score between 0 and 5 that scores how well the LLM answer matches the reference answer. In this case, they do not match perfectly, so the answer similarity score is not a perfect 5.
"""
logger.info("The answer similarity score is a score between 0 and 5 that scores how well the LLM answer matches the reference answer. In this case, they do not match perfectly, so the answer similarity score is not a perfect 5.")

answer_similarity_evaluator = AnswerSimilarityEvaluator()
score = answer_similarity_evaluator.evaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
logger.success(format_json(score))
score

"""
The answer consistency score is between 0.0 and 1.0, and measure whether the answer has information that does not appear in the retrieved context. In this case, the answer does appear in the retrieved context, so the score is 1.
"""
logger.info("The answer consistency score is between 0.0 and 1.0, and measure whether the answer has information that does not appear in the retrieved context. In this case, the answer does appear in the retrieved context, so the score is 1.")

answer_consistency_evaluator = AnswerConsistencyEvaluator()

score = answer_consistency_evaluator.evaluate(
    question, llm_answer, retrieved_context_list
)
logger.success(format_json(score))
score

"""
Augmentation accuracy measeures the percentage of the retrieved context that is in the answer. In this case, one of the retrieved contexts is in the answer, so this score is 0.5.
"""
logger.info("Augmentation accuracy measeures the percentage of the retrieved context that is in the answer. In this case, one of the retrieved contexts is in the answer, so this score is 0.5.")

augmentation_accuracy_evaluator = AugmentationAccuracyEvaluator()

score = augmentation_accuracy_evaluator.evaluate(
    question, llm_answer, retrieved_context_list
)
logger.success(format_json(score))
score

"""
Augmentation precision measures whether the relevant retrieved context makes it into the answer. Both of the retrieved contexts are relevant, but only one makes it into the answer. For that reason, this score is 0.5.
"""
logger.info("Augmentation precision measures whether the relevant retrieved context makes it into the answer. Both of the retrieved contexts are relevant, but only one makes it into the answer. For that reason, this score is 0.5.")

augmentation_precision_evaluator = AugmentationPrecisionEvaluator()

score = augmentation_precision_evaluator.evaluate(
    question, llm_answer, retrieved_context_list
)
logger.success(format_json(score))
score

"""
Retrieval precision measures the percentage of retrieved context is relevant to answer the question. In this case, both of the retrieved contexts are relevant to answer the question, so the score is 1.0.
"""
logger.info("Retrieval precision measures the percentage of retrieved context is relevant to answer the question. In this case, both of the retrieved contexts are relevant to answer the question, so the score is 1.0.")

retrieval_precision_evaluator = RetrievalPrecisionEvaluator()

score = retrieval_precision_evaluator.evaluate(
    question, llm_answer, retrieved_context_list
)
logger.success(format_json(score))
score

"""
The `TonicValidateEvaluator` can calculate all of Tonic Validate's metrics at once.
"""
logger.info(
    "The `TonicValidateEvaluator` can calculate all of Tonic Validate's metrics at once.")

tonic_validate_evaluator = TonicValidateEvaluator()

scores = tonic_validate_evaluator.evaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
logger.success(format_json(scores))

scores.score_dict

"""
You can also evaluate more than one query and response at once using `TonicValidateEvaluator`, and return a `tonic_validate` `Run` object that can be logged to the Tonic Validate UI (validate.tonic.ai).

To do this, you put the questions, LLM answers, retrieved context lists, and reference answers into lists and cal `evaluate_run`.
"""
logger.info("You can also evaluate more than one query and response at once using `TonicValidateEvaluator`, and return a `tonic_validate` `Run` object that can be logged to the Tonic Validate UI (validate.tonic.ai).")

tonic_validate_evaluator = TonicValidateEvaluator()

scores = tonic_validate_evaluator.evaluate_run(
    [question], [llm_answer], [retrieved_context_list], [reference_answer]
)
logger.success(format_json(scores))
scores.run_data[0].scores

"""
## Labelled RAG Dataset Example

Let's use the dataset `EvaluatingLlmSurveyPaperDataset` and evaluate the default LlamaIndex RAG system using Tonic Validate's answer similarity score. `EvaluatingLlmSurveyPaperDataset` is a `LabelledRagDataset`, so it contains reference correct answers for each question. The dataset contains 276 questions and reference answers about the paper *Evaluating Large Language Models: A Comprehensive Survey*.

We'll use `TonicValidateEvaluator` with the answer similarity score metric to evaluate the responses from the default RAG system on this dataset.
"""
logger.info("## Labelled RAG Dataset Example")

# !llamaindex-cli download-llamadataset EvaluatingLlmSurveyPaperDataset --download-dir ./data


rag_dataset = LabelledRagDataset.from_json(
    f"{os.path.dirname(__file__)}/data/rag_dataset.json")

documents = SimpleDirectoryReader(input_dir=f"{os.path.dirname(__file__)}/data/source_files").load_data(
    num_workers=4
)  # parallel loading

index = VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine()

predictions_dataset = rag_dataset.make_predictions_with(query_engine)

questions, retrieved_context_lists, reference_answers, llm_answers = zip(
    *[
        (e.query, e.reference_contexts, e.reference_answer, p.response)
        for e, p in zip(rag_dataset.examples, predictions_dataset.predictions)
    ]
)


tonic_validate_evaluator = TonicValidateEvaluator(
    metrics=[AnswerSimilarityMetric()], model_evaluator="gpt-4-1106-preview"
)

scores = tonic_validate_evaluator.evaluate_run(
    questions, retrieved_context_lists, reference_answers, llm_answers
)
logger.success(format_json(scores))

"""
The `overall_scores` gives the average score over the 276 questions in the dataset.
"""
logger.info(
    "The `overall_scores` gives the average score over the 276 questions in the dataset.")

scores.overall_scores

"""
Using `pandas` and `matplotlib`, we can plot a histogram of the similarity scores.
"""
logger.info(
    "Using `pandas` and `matplotlib`, we can plot a histogram of the similarity scores.")


score_list = [x.scores["answer_similarity"] for x in scores.run_data]
value_counts = pd.Series(score_list).value_counts()

fig, ax = plt.subplots()
ax.bar(list(value_counts.index), list(value_counts))
ax.set_title("Answer Similarity Score Value Counts")
plt.show()

"""
As 0 is the most common score, there is much room for improvement. This makes sense, as we are using the default parameters. We could imrpove these results by tuning the many possible RAG parameters to optimize this score.
"""
logger.info("As 0 is the most common score, there is much room for improvement. This makes sense, as we are using the default parameters. We could imrpove these results by tuning the many possible RAG parameters to optimize this score.")

logger.info("\n\n[DONE]", bright=True)
