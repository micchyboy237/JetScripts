import os

from jet.llm.ollama.base import Ollama
from tqdm import tqdm
from jet.actions.prompts_generator import PromptsGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.file.utils import save_file
from jet.transformers.object import make_serializable


def main():
    # data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    prompts = [
        "Tell me about yourself.",
        "She gave her friend a book and a pen.",
    ]

    processor = PromptsGenerator(llm=Ollama(model="llama3.1"))
    response_stream = processor.process(prompts)

    generation_results = []
    generation_tqdm = tqdm(response_stream, total=len(prompts))

    for tqdm_idx, (text, response) in enumerate(generation_tqdm):
        result = {
            "prompt": text,
            "results": response.data
        }
        generation_results.append(result)

        logger.info(f"DONE RESPONSE {tqdm_idx + 1}")
        # Update the progress bar after processing each node
        generation_tqdm.update(1)


if __name__ == "__main__":
    main()

    logger.info("\n\n[DONE]", bright=True)
