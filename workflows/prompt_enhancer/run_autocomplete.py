import os

from tqdm import tqdm
from jet.llm.main.autocomplete_prompts_generator import AutocompletePromptsGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.file.utils import save_file
from jet.transformers.object import make_serializable

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
OUTPUT_DIR = f"{GENERATED_DIR}/output"
OUTPUT_FILE = f"{OUTPUT_DIR}/generated-prompts.json"


def main():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    prompts = [
        "Tell me",
    ]

    processor = AutocompletePromptsGenerator(data_path)
    response_stream = processor.process(prompts)

    generation_results = []
    generation_tqdm = tqdm(response_stream, total=len(prompts))

    for tqdm_idx, (text, results) in enumerate(generation_tqdm):
        logger.newline()
        logger.info("RESULTS:")
        logger.success(format_json(results))

        generation_results.append({
            "prompt": text,
            "results": results
        })

        save_file(generation_results, OUTPUT_FILE)

        logger.info(f"DONE RESPONSE {tqdm_idx + 1}")
        # Update the progress bar after processing each node
        generation_tqdm.update(1)


if __name__ == "__main__":
    main()

    logger.info("\n\n[DONE]", bright=True)
