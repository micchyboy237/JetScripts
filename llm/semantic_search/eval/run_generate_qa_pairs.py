import json
import os
from jet.file.utils import save_file
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json


def main():
    query = "Tell me about yourself."
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    llm_model = "llama3.1"
    num_questions_per_chunk = 3

    file_no_ext = os.path.basename(__file__).split(".")[0]
    generated_dir = os.path.join("generated", file_no_ext)

    qa_dataset_generator = QADatasetGenerator(
        data_path=data_path,
        num_questions_per_chunk=num_questions_per_chunk,
        llm_model=llm_model,
    )

    qa_pairs_dataset = qa_dataset_generator.generate_qa_pairs()
    logger.newline()
    logger.info("Generated QA pairs dataset:")
    logger.success(format_json(qa_pairs_dataset))

    save_file(qa_pairs_dataset, os.path.join(
        generated_dir, "qa_pairs_dataset.json"))

    eval_top_k = 10
    evaluate_dataset_stream = qa_pairs_dataset.evaluate_dataset(
        top_k=eval_top_k)

    eval_results = []
    for idx, eval_result in enumerate(evaluate_dataset_stream):
        eval_results.append(eval_result)

        logger.newline()
        logger.info(f"Retrieval eval result {idx + 1}:")
        logger.success(format_json(eval_result))
        save_file(eval_results, os.path.join(
            generated_dir, "eval_results.json"))

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
