import json
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.llm.helpers.question_generator import QuestionGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable


def main():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    llm_model = "llama3.1"
    num_questions_per_chunk = 3

    questions_generator = QuestionGenerator(
        data_path=data_path,
        num_questions_per_chunk=num_questions_per_chunk,
        llm_model=llm_model,
    )

    questions = questions_generator.generate_questions()
    logger.newline()
    logger.info(f"Generated eval questions ({(len(questions))}):")
    logger.success(format_json(questions))

    eval_results = questions_generator.evaluate_questions(questions)
    logger.newline()
    logger.info("Evaluated questions results:")
    logger.success(format_json(eval_results))

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
