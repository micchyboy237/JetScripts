from jet.llm.main.prompts_generator import PromptsGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json


def main():
    # data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    prompt = "List your work history and achievements."

    processor = PromptsGenerator()
    response = processor.process(prompt=prompt)

    logger.newline()
    logger.info("RESPONSE:")
    logger.success(format_json(response))
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
