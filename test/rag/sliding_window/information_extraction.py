from typing import List, Dict
from utils import sliding_window_split, TextChunk
import mlx.core as mx
from mlx_lm import load, generate


def extract_information(text: str, entity_type: str, model_path: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125") -> List[Dict]:
    """Extract specific information from a large text using sliding windows."""
    chunks = sliding_window_split(text)
    model, tokenizer = load(model_path)
    results: List[Dict] = []

    for chunk in chunks:
        prompt = f"Extract all {entity_type} from the following text:\n{chunk['text']}"
        extracted = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        results.append(
            {"chunk": chunk["text"], "entities": extracted.strip().split(", ")})
    return results


if __name__ == "__main__":
    sample_text = "Apple released the iPhone 16. Google launched Pixel 9." * 100
    entities = extract_information(sample_text, "products")
    for result in entities:
        print(
            f"Chunk: {result['chunk'][:50]}... Entities: {result['entities']}")
