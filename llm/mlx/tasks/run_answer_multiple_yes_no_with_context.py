import os
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.answer_multiple_yes_no_with_context import QuestionContext, answer_multiple_yes_no_with_context, AnswerResult
from jet.models.model_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.vectors.document_types import HeaderDocument

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    query = "List all ongoing and upcoming isekai anime 2025."
    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    doc_texts = [doc.text for doc in docs]

    question = f"Does this contain actual data that answer this browser query: \"{query}\"?"
    contexts = doc_texts
    question_contexts: list[QuestionContext] = [
        {"question": question, "context": ctx} for ctx in contexts
    ]
    results = answer_multiple_yes_no_with_context(
        question_contexts, model_path=model_path
    )
    logger.gray("Results:")
    logger.success(format_json(results))
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    save_file(results, f"{output_dir}/results.json")
