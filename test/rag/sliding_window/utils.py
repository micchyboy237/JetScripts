from typing import List, TypedDict

class TextChunk(TypedDict):
    text: str
    start_idx: int
    end_idx: int

def sliding_window_split(text: str, window_size: int = 500, stride: int = 250) -> List[TextChunk]:
    """Split text into overlapping chunks using a sliding window."""
    chunks: List[TextChunk] = []
    for i in range(0, len(text), stride):
        end_idx = min(i + window_size, len(text))
        chunks.append({"text": text[i:end_idx], "start_idx": i, "end_idx": end_idx})
        if end_idx == len(text):
            break
    return chunks