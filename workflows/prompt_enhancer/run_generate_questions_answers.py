from jet.actions.intervew_qa_generator import InterviewQAGenerator
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

    processor = InterviewQAGenerator()
    response = processor.process(data_path=data_path)
    formatted_response = "\n\n".join([
        f"Question: {item.question}\nAnswer: {item.answer}"
        for item in response.data
    ])

    logger.newline()
    logger.info("RESPONSE:")
    logger.success(formatted_response)
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
