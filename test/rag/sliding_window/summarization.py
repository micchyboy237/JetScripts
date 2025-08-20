from typing import List
from utils import sliding_window_split, TextChunk
import mlx.core as mx
from mlx_lm import load, generate


def summarize_document(text: str, model_path: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125") -> str:
    """Summarize a long document using sliding windows and an LLM."""
    model, tokenizer = load(model_path)
    chunks = sliding_window_split(text)
    summaries: List[str] = []

    for chunk in chunks:
        prompt = f"Summarize the following text in 1-2 sentences:\n{chunk['text']}"
        summary = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        summaries.append(summary.strip())

    # Combine chunk summaries into a final summary
    combined_prompt = f"Combine the following summaries into a concise overall summary:\n{' '.join(summaries)}"
    final_summary = generate(
        model, tokenizer, prompt=combined_prompt, max_tokens=200)
    return final_summary.strip()


if __name__ == "__main__":
    sample_text = "This is a long research paper about AI advancements. " * 1000
    summary = summarize_document(sample_text)
    print("Summary:", summary)
