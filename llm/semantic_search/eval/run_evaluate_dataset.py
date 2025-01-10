import json
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.logger import logger
from jet.transformers.object import make_serializable


def main():
    query = "Tell me about yourself."
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    llm_model = "llama3.1"
    num_questions_per_chunk = 3

    qa_dataset_generator = QADatasetGenerator(
        data_path=data_path,
        num_questions_per_chunk=num_questions_per_chunk,
        llm_model=llm_model,
    )

    questions_rag_dataset = qa_dataset_generator.generate_dataset()
    logger.newline()
    logger.info("Generated questions RAG dataset:")
    logger.success(json.dumps(make_serializable(
        questions_rag_dataset), indent=2))

    # eval_question_results = questions_rag_dataset.evaluate_question(query)
    # logger.info("Evaluated question results:")
    # logger.success(json.dumps(make_serializable(
    #     eval_question_results), indent=2))

    # eval_queries_results = questions_rag_dataset.evaluate_queries()
    # logger.info(f"Evaluated queries results ({len(eval_queries_results)}):")
    # logger.success(json.dumps(make_serializable(
    #     eval_queries_results), indent=2))

    eval_qa_dataset = []
    eval_qa_dataset_stream = questions_rag_dataset.evaluate_qa_dataset()
    for idx, result in enumerate(eval_qa_dataset_stream):
        eval_qa_dataset.append(result)
        logger.newline()
        logger.debug(f"Eval QA result {idx + 1}:")
        logger.success(json.dumps(make_serializable(result), indent=2))

    logger.newline()
    logger.success(f"Evaluated QA dataset results: ({
                   len(eval_queries_results)})")

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
