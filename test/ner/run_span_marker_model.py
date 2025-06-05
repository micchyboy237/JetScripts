from jet.logger import logger
from transformers import AutoTokenizer
import transformers
import os
from typing import List, Dict, Any
from pydantic import BaseModel
from span_marker import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "true"

print("transformers:", transformers.__version__)


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float

    def __str__(self) -> str:
        return self.text


def chunk_text(
    text: str,
    tokenizer: SpanMarkerTokenizer,
    max_length: int = 512,
    stride: int = 128
) -> List[Dict[str, Any]]:
    """
    Split text into chunks with overlap, preserving character indices.

    Args:
        text: Input text to chunk.
        tokenizer: SpanMarker tokenizer.
        max_length: Maximum token length per chunk (including special tokens).
        stride: Overlap size in tokens.

    Returns:
        List of dictionaries with chunk text and character offsets.
    """
    try:
        # Use the SpanMarkerTokenizer directly, which supports encode_plus
        tokens = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            return_tensors=None,
            add_special_tokens=True,
            max_length=None,  # No truncation here, we handle it manually
            return_attention_mask=False
        )
        input_ids = tokens["input_ids"]
        offset_mapping = tokens["offset_mapping"]

        chunks = []
        for i in range(0, len(input_ids), max_length - stride):
            start_idx = i
            end_idx = min(i + max_length, len(input_ids))
            chunk_input_ids = input_ids[start_idx:end_idx]
            chunk_offsets = offset_mapping[start_idx:end_idx]

            # Calculate character offsets for the chunk
            char_start = chunk_offsets[0][0] if chunk_offsets else 0
            char_end = chunk_offsets[-1][1] if chunk_offsets else len(text)

            # Decode tokens back to text
            chunk_text = tokenizer.decode(
                chunk_input_ids, skip_special_tokens=True)

            chunks.append({
                "text": chunk_text,
                "input_ids": chunk_input_ids,
                "offset_mapping": chunk_offsets,
                "char_start": char_start,
                "char_end": char_end
            })

        return chunks
    except Exception as e:
        logger.error(f"Failed to chunk text: {e}")
        raise


def merge_entities(
    entities: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge entity predictions from chunks, resolving overlaps by highest score.

    Args:
        entities: List of entity predictions from all chunks.
        chunks: List of chunk metadata.

    Returns:
        Deduplicated list of entities with adjusted character indices.
    """
    merged = []
    seen_spans = {}

    for chunk_idx, chunk_entities in enumerate(entities):
        char_start_offset = chunks[chunk_idx]["char_start"]
        for entity in chunk_entities:
            # Adjust character indices to original text
            adjusted_entity = {
                "span": entity["span"],
                "label": entity["label"],
                "score": entity["score"],
                "char_start_index": entity["char_start_index"] + char_start_offset,
                "char_end_index": entity["char_end_index"] + char_start_offset
            }
            span_key = (adjusted_entity["char_start_index"],
                        adjusted_entity["char_end_index"])

            # Keep entity with higher score if span overlaps
            if span_key not in seen_spans or adjusted_entity["score"] > seen_spans[span_key]["score"]:
                seen_spans[span_key] = adjusted_entity

    return list(seen_spans.values())


def process_long_text(
    text: str,
    model: SpanMarkerModel,
    tokenizer: SpanMarkerTokenizer,
    max_length: int = 512,
    stride: int = 128
) -> List[SpanMarkerWord]:
    """
    Process long text with NER, handling token length limits.

    Args:
        text: Input text.
        model: SpanMarker model for NER.
        tokenizer: SpanMarker tokenizer.
        max_length: Maximum token length per chunk.
        stride: Overlap size in tokens.

    Returns:
        List of SpanMarkerWord objects with entity information.
    """
    logger.info("Chunking text...")
    chunks = chunk_text(text, tokenizer, max_length, stride)
    logger.debug(f"Created {len(chunks)} chunks")

    all_entities = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Predicting entities for chunk {i+1}/{len(chunks)}...")
        try:
            chunk_entities = model.predict(chunk["text"])
            all_entities.append(chunk_entities)
        except Exception as e:
            logger.error(f"Failed to predict entities for chunk {i+1}: {e}")
            raise

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

    return results


# Example usage
if __name__ == "__main__":
    try:
        model = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-bert-base-fewnerd-fine-super")
        tokenizer = model.tokenizer

        # Example long text
        long_text = (
            "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. "
            "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. "
            * 10  # Simulate long text by repeating
        )

        results = process_long_text(long_text, model, tokenizer)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
