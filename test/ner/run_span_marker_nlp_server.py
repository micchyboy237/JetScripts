from jet.logger import logger
import spacy
from spacy import displacy
from spacy.tokens.doc import Doc

# Simulated logger for demonstration (replace with jet.logger if available)


def process_text(text: str, nlp: spacy.language.Language) -> Doc:
    """Process text with spaCy pipeline."""
    return nlp(text)


def log_entities(doc: Doc) -> None:
    """Log named entities with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Entities ({len(doc.ents)}):")
    for entity in doc.ents:
        logger.newline()
        logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
        logger.log("Label:", entity.label_, colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity.start_char}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("End:", f"{entity.end_char}", colors=["WHITE", "SUCCESS"])
        # SpanMarker provides scores via token attributes or pipeline context
        # SpanMarker stores score in entity
        score = getattr(entity, "score", "No score")
        logger.log("Score:", f"{score}", colors=["WHITE", "SUCCESS"])
        logger.log("Vector Norm:", f"{entity.vector_norm if entity.has_vector else 'No vector'}",
                   colors=["WHITE", "INFO"])
        logger.log("---")


def log_noun_chunks(doc: Doc) -> None:
    """Log noun chunks with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Noun Chunks ({len(list(doc.noun_chunks))}):")
    for chunk in doc.noun_chunks:
        logger.newline()
        logger.log("Text:", chunk.text, colors=["WHITE", "INFO"])
        logger.log("Root Text:", chunk.root.text, colors=["WHITE", "INFO"])
        logger.log("Root Dependency:", chunk.root.dep_,
                   colors=["WHITE", "SUCCESS"])
        logger.log("Root Head Text:", chunk.root.head.text,
                   colors=["WHITE", "SUCCESS"])
        logger.log("---")


def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Add SpanMarker as an additional NER pipeline
    nlp.add_pipe("span_marker", config={
        "model": "tomaarsen/span-marker-mbert-base-multinerd"
    }, last=True)

    # Input text
    text = "##### Headhunted to Another World: From Salaryman to Big Four!\nIsekai\nFantasy\nComedy\nRelease Date : January 1, 2025\nRelease Date\n: January 1, 2025\nJapanese Title : Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio : Geek Toys, CompTown Based On : Manga Creator : Benigashira Streaming Service(s) : Crunchyroll\nJapanese Title\n: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi\n\nStudio\n: Geek Toys, CompTown\n\nBased On\n: Manga\n\nCreator\n: Benigashira\n\nStreaming Service(s)\n: Crunchyroll\nPowered by\nExpand Collapse\nPlenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.\nRelated"

    # Process text
    doc = process_text(text, nlp)

    # Log entities and noun chunks
    log_entities(doc)
    log_noun_chunks(doc)

    # Visualize entities
    displacy.serve(doc, style="ent", port=5002, options={"colors": {
                   "PER": "#ff9999", "LOC": "#99ff99", "DATE": "#9999ff"}})


if __name__ == "__main__":
    main()
