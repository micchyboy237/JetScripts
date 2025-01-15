import asyncio
import json
import os

from jet.data.utils import generate_unique_hash
from jet.llm.extractors.interview_qa_extractor import InterviewQAExtractor
from tqdm import tqdm
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from jet.vectors.metadata import generate_metadata, parse_nodes
from llama_index.core.extractors.metadata_extractors import QuestionsAnsweredExtractor, SummaryExtractor
from llama_index.core.readers.file.base import SimpleDirectoryReader
llm_settings = initialize_ollama_settings()

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
OUTPUT_DIR = F"{GENERATED_DIR}/output"


def load_metadata_dicts(path):
    logger.debug(f"Loading file from: {path}")
    with open(path, "r") as fp:
        return json.load(fp)


def save_metadata_dicts(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        # Load existing data
        current_data: list = load_metadata_dicts(path)
    except:
        # If file does not exist or there's an issue, initialize as an empty list
        current_data = []

    # Combine current data and new data
    combined_data = current_data + data

    # Ensure uniqueness by 'node_id'
    unique_data = {item["node_id"]: item for item in combined_data}.values()

    with open(path, "w") as fp:
        json.dump(list(unique_data), fp, indent=2, ensure_ascii=False)

    logger.success("Saved file to: " + path, bright=True)


async def main():
    chunk_size = 1024
    chunk_overlap = 20

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    logger.newline()
    logger.info("Loading data...")
    docs = SimpleDirectoryReader(data_dir).load_data(show_progress=True)
    logger.log("All docs:", len(docs), colors=["DEBUG", "SUCCESS"])
    base_nodes = parse_nodes(docs, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap)

    # Metadata generation
    logger.newline()
    logger.info("Metadata generation...")
    summary_extractor = SummaryExtractor(llm=llm_settings.llm, summaries=[
        "self"], show_progress=True)
    # questions_answered_extractor = QuestionsAnsweredExtractor(
    #     llm=llm_settings.llm, questions=5, show_progress=True)
    questions_answered_extractor = InterviewQAExtractor(llm=llm_settings.llm)
    extractors = [summary_extractor, questions_answered_extractor]

    metadata_results: list = []
    metadata_tqdm = tqdm(total=len(base_nodes))
    async for node, metadata in generate_metadata(base_nodes, extractors):
        logger.newline()
        logger.info(f"Node ID: {node.node_id}")
        logger.debug(f"Text length: {len(node.text)}")
        logger.log("Metadata:",
                   format_json(metadata_results), colors=["DEBUG", "SUCCESS"])
        metadata_results.append({
            "node_id": generate_unique_hash(),
            "metadata": metadata
        })
        save_metadata_dicts(
            f"{OUTPUT_DIR}/jet-resume-metadata.json", metadata_results)

        # Update the progress bar after processing each node
        metadata_tqdm.update(1)

if __name__ == "__main__":
    asyncio.run(main())
