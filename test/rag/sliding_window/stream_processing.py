from typing import List, Optional
from utils import sliding_window_split, TextChunk
import mlx.core as mx
from mlx_lm import load, generate


class StreamProcessor:
    def __init__(self, window_size: int = 500, stride: int = 250, model_path: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"):
        self.window_size = window_size
        self.stride = stride
        self.buffer = ""
        self.model, self.tokenizer = load(model_path)

    def process_stream(self, new_text: str) -> Optional[str]:
        """Process incoming text incrementally and generate a summary."""
        self.buffer += new_text
        if len(self.buffer) < self.window_size:
            return None

        chunks = sliding_window_split(
            self.buffer, self.window_size, self.stride)
        latest_chunk = chunks[-1]
        prompt = f"Summarize the following text in 1 sentence:\n{latest_chunk['text']}"
        summary = generate(self.model, self.tokenizer,
                           prompt=prompt, max_tokens=100)

        # Slide the buffer
        self.buffer = self.buffer[self.stride:]
        return summary.strip()


if __name__ == "__main__":
    processor = StreamProcessor()
    stream_texts = ["AI is transforming industries. " *
                    50, "Machine learning is key. " * 50]
    for text in stream_texts:
        summary = processor.process_stream(text)
        if summary:
            print("Stream Summary:", summary)
