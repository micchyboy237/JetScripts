import os
import shutil
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.libs.stanza.pipeline import StanzaPipelineCache
from jet.logger import logger
from tqdm import tqdm

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def initialize_pipeline() -> StanzaPipelineCache:
    """Initialize Stanza pipeline with all processors."""
    stanza_pipeline = StanzaPipelineCache(use_gpu=True)
    return stanza_pipeline

def extract_nlp(text: str) -> dict:
    """Run all Stanza extraction methods and return full results."""
    logger.info("Running full Stanza pipeline extractions")
    stanza_pipeline = initialize_pipeline()
    pos = stanza_pipeline.extract_pos(text)
    sentences = stanza_pipeline.extract_sentences(text)
    entities = stanza_pipeline.extract_entities(text)
    dependencies = stanza_pipeline.extract_dependencies(text)
    constituencies = stanza_pipeline.extract_constituencies(text)
    scenes = stanza_pipeline.extract_scenes(text)

    # Also, fetch the underlying stanza.Document and construct detailed token info per sentence
    doc = stanza_pipeline._pipeline(text)
    sentence_details = []
    for sent in doc.sentences:
        sentence_details.append(sent.to_dict())

    return {
        "pos": pos,
        "sentences": sentences,
        "entities": entities,
        "dependencies": dependencies,
        "constituencies": constituencies,
        "scenes": scenes,
        "sentence_details": sentence_details,
    }

if __name__ == "__main__":
    model = "embeddinggemma"
    chunks = load_sample_data(model=model)
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks...")):
        results = extract_nlp(chunk)
        for key, nlp_results in results.items():
            save_file(nlp_results, f"{OUTPUT_DIR}/chunk_{chunk_idx + 1}/{key}_results.json")
