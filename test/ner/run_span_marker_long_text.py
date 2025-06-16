from jet.models.embeddings.base import get_embedding_function
from jet.llm.embeddings.fast_embedding import EmbeddingGenerator
from jet.logger import logger
from jet.file.utils import load_file, save_file
import os
from typing import List, Dict, Any
from pydantic import BaseModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set environment variables for Mac M1 compatibility
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float

    def __str__(self) -> str:
        return self.text


def merge_entities(
    entities: List[List[Dict[str, Any]]],
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    merged = []
    seen_spans = {}
    for chunk_idx, chunk_entities in enumerate(entities):
        char_start_offset = chunks[chunk_idx]["char_start"]
        for entity in chunk_entities:
            adjusted_entity = {
                "span": entity["span"],
                "label": entity["label"],
                "score": entity["score"],
                "char_start_index": entity["char_start_index"] + char_start_offset,
                "char_end_index": entity["char_end_index"] + char_start_offset
            }
            span_key = (adjusted_entity["char_start_index"],
                        adjusted_entity["char_end_index"])
            if span_key not in seen_spans or adjusted_entity["score"] > seen_spans[span_key]["score"]:
                seen_spans[span_key] = adjusted_entity
    return list(seen_spans.values())


def process_long_text(
    text: str,
    model_name: str = "tomaarsen/span-marker-bert-base-fewnerd-fine-super",
    max_length: int = 256,
    stride: int = 64,
    batch_size: int = 32,
    use_mps: bool = True
) -> List[SpanMarkerWord]:
    try:
        logger.info("Loading model...")
        generator = EmbeddingGenerator(
            model_name, model_type="span_marker", use_mps=use_mps
        )
        logger.info("Chunking text...")
        chunks = generator.chunk_text(
            text, max_length=max_length, stride=stride)
        logger.debug(f"Created {len(chunks)} chunks")

        logger.info("Processing chunks in batches...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embed_func = get_embedding_function(model_name)
        # all_entities = generator.generate_embeddings(
        #     chunk_texts,
        #     batch_size=batch_size,
        #     max_length=max_length,
        #     normalize=False
        # )
        all_entities = embed_func(chunk_texts)

        logger.debug("Merging entities...")
        merged_entities = merge_entities(all_entities, chunks)

        results = [
            SpanMarkerWord(
                text=entity["span"],
                lemma=entity["label"],
                score=entity["score"],
                start_idx=entity["char_start_index"],
                end_idx=entity["char_end_index"]
            )
            for entity in merged_entities
        ]

        logger.debug("Extracted Entities:")
        for result in results:
            logger.info(f"Text: {result.text}")
            logger.info(f"Lemma: {result.lemma}")
            logger.info(f"Score: {result.score:.4f}")
            logger.info(f"Start: {result.start_idx}")
            logger.info(f"End: {result.end_idx}")
            logger.info("---")

        return chunk_texts, results
    except Exception as e:
        logger.error(f"Error in processing text: {e}")
        raise


if __name__ == "__main__":
    try:
        data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/context.md"
        long_text = load_file(data_path)
        chunk_texts, results = process_long_text(long_text)
        output_dir = os.path.join(os.path.dirname(
            __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
        save_file(chunk_texts, f"{output_dir}/chunks.json")
        save_file(results, f"{output_dir}/results.json")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
