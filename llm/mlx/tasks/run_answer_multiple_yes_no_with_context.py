import os
from jet.file.utils import save_file
from jet.llm.mlx.tasks.answer_multiple_yes_no_with_context import QuestionContext, answer_multiple_yes_no_with_context, AnswerResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    question = "Which planet in our solar system has one or more moons?"
    contexts = [
        "Venus is the second planet from the Sun and has no natural moons.",
        "Jupiter is the largest planet and has at least 79 known moons, including Ganymede.",
        "Mars has two small moons named Phobos and Deimos.",
        "Saturn is known for its rings and has 83 moons with confirmed orbits."
    ]
    question_contexts: list[QuestionContext] = [
        {"question": question, "context": ctx} for ctx in contexts
    ]
    results = answer_multiple_yes_no_with_context(
        question_contexts, model_path="llama-3.2-3b-instruct-4bit"
    )
    logger.gray("Results:")
    logger.success(format_json(results))
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    save_file(results, f"{output_dir}/results.json")
