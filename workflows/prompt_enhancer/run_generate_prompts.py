import os

from tqdm import tqdm
from jet.llm.main.prompts_generator import PromptsGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.file.utils import save_file
from jet.transformers.object import make_serializable

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
OUTPUT_DIR = f"{GENERATED_DIR}/output"
OUTPUT_FILE = f"{OUTPUT_DIR}/generated-prompts.json"


def main():
    # data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    prompts = [
        "Tell me about yourself."
        "She gave her friend a book and a pen.",
        "The teacher explained the lesson and the homework assignment.",
        "We bought apples, oranges, and bananas at the market.",
        "He emailed the report and sent the invitation to all the team members.",
        "They played soccer, basketball, and tennis during the weekend."
        "She packed her suitcase and backpack for the trip.",
        "He cooked dinner and prepared dessert for the guests.",
        "They painted the fence and the shed in the backyard.",
        "I bought a pair of shoes and a jacket at the store.",
        "The children drew pictures and colored them during class.",
        "We built a sandcastle and a moat at the beach.",
        "She arranged the flowers and placed the vase on the table.",
        "He sold the car and the motorcycle to a collector.",
        "They took the book and the magazine from the shelf.",
        "I borrowed a pencil and an eraser from my friend.",
    ]

    processor = PromptsGenerator()
    response_stream = processor.process(prompts)

    generation_results = []
    generation_tqdm = tqdm(response_stream, total=len(prompts))

    for tqdm_idx, (text, response) in enumerate(generation_tqdm):
        logger.newline()
        logger.info("RESPONSE:")
        logger.success(format_json(response))

        generation_results.append({
            "prompt": text,
            "results": response.data
        })

        save_file(generation_results, OUTPUT_FILE)

        logger.info(f"DONE RESPONSE {tqdm_idx + 1}")
        # Update the progress bar after processing each node
        generation_tqdm.update(1)


if __name__ == "__main__":
    main()

    logger.info("\n\n[DONE]", bright=True)
