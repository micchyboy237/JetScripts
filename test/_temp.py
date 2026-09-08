from chonkie.chunker.code import CodeChunker

cc = CodeChunker(tokenizer="character", chunk_size=512, language="auto")
code = """
import random, time

def backoff(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, delay * 0.1)
"""
cc.chunk(code)  # Does this hang?
