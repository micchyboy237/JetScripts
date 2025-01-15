import os

from tqdm import tqdm
from jet.file.utils import save_file
from jet.llm.main.sources_generator import SourcesGenerator
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
OUTPUT_DIR = f"{GENERATED_DIR}/output"
OUTPUT_FILE = f"{OUTPUT_DIR}/jet-resume-sources.json"

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"


def main():
    processor = SourcesGenerator(DATA_DIR)
    response_stream = processor.process()

    sources_dict: dict = {}
    generation_results_dict: dict = {
        "data_dir": os.path.realpath(DATA_DIR),
        "data": sources_dict
    }
    generation_tqdm = tqdm(response_stream, total=len(processor.nodes))

    for tqdm_idx, response in enumerate(generation_tqdm):
        response_dict = make_serializable(response)
        new_sources = response_dict.get("data", [])

        for item_idx, item in enumerate(response.data):
            file_path = os.path.join(DATA_DIR, item.filename)
            with open(file_path) as file:
                # Read the entire content of the file
                file_content = file.read()
            file_content_lines = file_content.splitlines()
            num_lines = list(set(item.lines))
            num_lines.sort()
            source_line_indexes = [line - 1 for line in num_lines]
            start_index = source_line_indexes[0]
            end_index = source_line_indexes[-1] if len(
                source_line_indexes) > 1 else start_index + 1
            source_lines = file_content_lines[start_index:end_index]
            source_lines = [line for line in source_lines if line.strip()]

            new_item = new_sources[item_idx]
            new_item.pop("filename")
            new_item["lines"] = num_lines
            new_item["sources"] = source_lines
            # new_item["file_path"] = file_path

            source_results_dict = sources_dict.get(item.filename, {
                "file_path": file_path,
                "sources": []
            })
            source_results: list = source_results_dict.get("sources", [])
            source_results.append(new_item)
            source_results_dict["sources"] = source_results
            sources_dict[item.filename] = source_results_dict

        logger.newline()
        logger.info(f"DONE RESPONSE {tqdm_idx + 1}:")
        logger.success(format_json(new_sources))

        formatted_sources = []
        for filename, source in generation_results_dict["data"].items():
            formatted_sources.append({
                "file_name": filename,
                **source
            })
        formatted_results_dict = {
            **generation_results_dict,
            "data": formatted_sources,
        }
        save_file(formatted_results_dict, OUTPUT_FILE)

        # Update the progress bar after processing each node
        generation_tqdm.update(1)

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
