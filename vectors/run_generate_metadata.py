import asyncio
import json
import os
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from jet.vectors.metadata import generate_metadata, parse_nodes
from llama_index.core.extractors.metadata_extractors import QuestionsAnsweredExtractor, SummaryExtractor
from llama_index.core.readers.file.base import SimpleDirectoryReader
llm_settings = initialize_ollama_settings()

GENERATED_DIR = os.path.join(
    "generated/" + os.path.splitext(os.path.basename(__file__))[0])
OUTPUT_DIR = F"{GENERATED_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_metadata_dicts(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(make_serializable(data), fp, indent=2, ensure_ascii=False)
    logger.success("Saved file to" + path, bright=True)


async def main():
    chunk_size = 1024
    chunk_overlap = 200

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
    questions_answered_extractor = QuestionsAnsweredExtractor(
        llm=llm_settings.llm, questions=5, show_progress=True)
    extractors = [summary_extractor, questions_answered_extractor]

    metadata_results: list = []
    async for node, metadata in generate_metadata(base_nodes, extractors):
        logger.newline()
        logger.info(f"Node ID: {node.node_id}")
        logger.debug(f"Text length: {len(node.text)}")
        logger.log("Metadata:",
                   format_json(metadata), colors=["DEBUG", "SUCCESS"])
        metadata_results.append({
            "node_id": node.node_id,
            "metadata": metadata
        })
        save_metadata_dicts(
            f"{OUTPUT_DIR}/llama2_metadata.json", metadata_results)

if __name__ == "__main__":
    asyncio.run(main())
