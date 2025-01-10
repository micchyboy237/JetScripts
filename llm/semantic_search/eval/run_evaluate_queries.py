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

    eval_queries_results = questions_rag_dataset.evaluate_queries()
    logger.info(f"Evaluated queries results ({len(eval_queries_results)}):")
    logger.success(json.dumps(make_serializable(
        eval_queries_results), indent=2))

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
