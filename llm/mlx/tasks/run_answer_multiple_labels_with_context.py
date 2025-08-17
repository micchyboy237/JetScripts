import os
from jet.file.utils import save_file
from jet.llm.mlx.tasks.answer_multiple_labels_with_context import QuestionContext, answer_multiple_labels_with_context, AnswerResult
from jet.models.model_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    question = "What is the sentiment of the review?"
    contexts = [
        "The movie was a thrilling adventure with stunning visuals.",
        "The product failed to meet expectations and broke quickly.",
        "The service was adequate but unremarkable."
    ]
    question_contexts: list[QuestionContext] = [
        {"question": question, "context": ctx} for ctx in contexts
    ]
    labels = ["Positive", "Negative", "Neutral"]
    results = answer_multiple_labels_with_context(
        question_contexts, model_path="llama-3.2-3b-instruct-4bit", labels=labels
    )
    logger.gray("Results:")
    logger.success(format_json(results))
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    save_file(results, f"{output_dir}/results.json")
