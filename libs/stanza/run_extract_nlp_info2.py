import os
import shutil
from jet.libs.stanza.rag_stanza import StanzaPipelineCache
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

text = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
)

# Define the default set of processors for most complete English analyses
ALL_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency"


def initialize_pipeline(processors: str = ALL_PROCESSORS, lang: str = "en") -> StanzaPipelineCache:
    """Initialize Stanza pipeline with all processors."""
    stanza_pipeline = StanzaPipelineCache(
        lang=lang,
        processors=processors,
        use_gpu=True,
        verbose=True
    )
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

def main():
    """Run all processor examples on each document sequentially and save results with progress tracking."""
    results = extract_nlp(text)
    for key, nlp_results in results.items():
        save_file(nlp_results, f"{OUTPUT_DIR}/{key}_results.json")

if __name__ == "__main__":
    main()