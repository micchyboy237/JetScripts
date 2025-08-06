import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def preprocess_text(text: str) -> str:
    """
    Preprocesses a single text by normalizing whitespace, converting to lowercase,
    and removing special characters.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, keeping alphanumeric and spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace (replace multiple spaces with single space, strip)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # dimensions = 512
    # model_name: EmbedModelType = "mxbai-embed-large"
    # model_name: EmbedModelType = "nomic-embed-text"
    model_name: EmbedModelType = "all-MiniLM-L6-v2"
    # Same example queries
    queries = [
        "##### The Water Magician\nAnime\nIsekai Action Fantasy\nRelease Date\nJuly 3, 2025\nJapanese Title\nMizu Zokusei no Mahoutsukai\nStudio\nTyphoon Graphics, WonderLand\nBased On\nNovel & Light Novel\nCreator\nTadashi Kudou\nStreaming Service(s)\nCrunchyroll\nPowered by Expand Collapse\nThe Water Magician is a light novel series that spawned a manga in 2021, which is still going on today. The anime debuted in July 2025, bringing forth another story that features a relatively overpowered protagonist. There is nothing wrong with that as there is a desire for OP MCs, but this anime's mileage will vary considerably from viewer to viewer.\nRyou gets a second go at life, and he not only picks up water magic but also has a trait that lets him remain endlessly young. ",
    ]
    # Same sample documents
    sample_docs = [
        "##### The Water Magician\nRyou gets a second go at life, and he not only picks up water magic but also has a trait that lets him remain endlessly young. The Water Magician has Ryou spend a bit of time on his own at first, and he does come across as a bit socially awkward at times. The story also does something entirely unexpected: Ryou's first friend and companion is a guy! Yes, he doesn't instantly summon a harem. ",
    ]
    # Preprocess queries and sample_docs
    queries = [preprocess_text(q) for q in queries]
    sample_docs = [preprocess_text(d) for d in sample_docs]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
