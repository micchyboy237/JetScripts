import os
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.answer_multiple_labels_with_context import QuestionContext, answer_multiple_labels_with_context, AnswerResult
from jet.models.model_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)

    question = "List trending isekai reincarnation anime this year."
    contexts = [
        header["text"]
        for header in headers
        if header["header_level"] != 1
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
