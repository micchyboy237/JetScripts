%pip install llama-index-evaluation-tonic-validateimport json

import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.evaluation.tonic_validate import (
    AnswerConsistencyEvaluator,
    AnswerSimilarityEvaluator,
    AugmentationAccuracyEvaluator,
    AugmentationPrecisionEvaluator,
    RetrievalPrecisionEvaluator,
    TonicValidateEvaluator,
)
question = "What makes Sam Altman a good founder?"
reference_answer = "He is smart and has a great force of will."
llm_answer = "He is a good founder because he is smart."
retrieved_context_list = [
    "Sam Altman is a good founder. He is very smart.",
    "What makes Sam Altman such a good founder is his great force of will.",
]
answer_similarity_evaluator = AnswerSimilarityEvaluator()
score = await answer_similarity_evaluator.aevaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
score
answer_consistency_evaluator = AnswerConsistencyEvaluator()

score = await answer_consistency_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
score
augmentation_accuracy_evaluator = AugmentationAccuracyEvaluator()

score = await augmentation_accuracy_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
score
augmentation_precision_evaluator = AugmentationPrecisionEvaluator()

score = await augmentation_precision_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
score
retrieval_precision_evaluator = RetrievalPrecisionEvaluator()

score = await retrieval_precision_evaluator.aevaluate(
    question, llm_answer, retrieved_context_list
)
score
tonic_validate_evaluator = TonicValidateEvaluator()

scores = await tonic_validate_evaluator.aevaluate(
    question,
    llm_answer,
    retrieved_context_list,
    reference_response=reference_answer,
)
scores.score_dict
tonic_validate_evaluator = TonicValidateEvaluator()

scores = await tonic_validate_evaluator.aevaluate_run(
    [question], [llm_answer], [retrieved_context_list], [reference_answer]
)
scores.run_data[0].scores
!llamaindex-cli download-llamadataset EvaluatingLlmSurveyPaperDataset --download-dir ./data
from llama_index.core import SimpleDirectoryReader

from llama_index.core.llama_dataset import LabelledRagDataset

from llama_index.core import VectorStoreIndex


rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")

documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data(
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
from tonic_validate.metrics import AnswerSimilarityMetric

tonic_validate_evaluator = TonicValidateEvaluator(
    metrics=[AnswerSimilarityMetric()], model_evaluator="gpt-4-1106-preview"
)

scores = await tonic_validate_evaluator.aevaluate_run(
    questions, retrieved_context_lists, reference_answers, llm_answers
)
scores.overall_scores
import matplotlib.pyplot as plt
import pandas as pd

score_list = [x.scores["answer_similarity"] for x in scores.run_data]
value_counts = pd.Series(score_list).value_counts()

fig, ax = plt.subplots()
ax.bar(list(value_counts.index), list(value_counts))
ax.set_title("Answer Similarity Score Value Counts")
plt.show()
