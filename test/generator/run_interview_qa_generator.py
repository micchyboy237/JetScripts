from jet.llm.main.intervew_qa_generator import InterviewQAGenerator
from jet.logger import logger


def main():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    query = """Generate real-world diverse questions and answers that an employer can have for a job interview based on provided context and schema.
    Example response format:
    {
        "data": [
            {
                "question": "Question 1",
                "answer": "Answer 1"
            }
        ]
    }
    """.strip()

    processor = InterviewQAGenerator(data_path)
    response = processor.process(query)

    logger.newline()
    logger.info("RESPONSE:")
    logger.success(format_json(response))
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
