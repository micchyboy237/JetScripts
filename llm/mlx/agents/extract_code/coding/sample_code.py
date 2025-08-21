def chunk_text(text: str, chunk_size: int) -> list:
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_chunks(chunks: list) -> list:
    """Process each chunk (e.g., add prefix)."""
    return [f"Chunk_{i}: {chunk}" for i, chunk in enumerate(chunks)]