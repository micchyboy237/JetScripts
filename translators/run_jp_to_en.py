# translator.py
import logging
from typing import List, Dict, Any, Iterator
import torch
from transformers import pipeline
from tqdm.auto import tqdm

# Configure root logger once (you can adjust level as needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def create_translator(
    model_name: str = "Helsinki-NLP/opus-mt-ja-en",
    device: str = "auto",
    batch_size: int = 32
) -> pipeline:
    """
    Create a batched translation pipeline with optimal device selection.
    
    Args:
        model_name: Hugging Face model ID
        device: "auto" (recommended), "mps", "cuda", or "cpu"
        batch_size: Larger = faster on MPS/GPU, smaller = lower memory
    
    Returns:
        Initialized pipeline with batched inference enabled
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            selected = "mps"
        elif torch.cuda.is_available():
            selected = "cuda"
        else:
            selected = "cpu"
    else:
        selected = device

    log.info(f"Loading model '{model_name}' on device: {selected.upper()} (batch_size={batch_size})")
    
    return pipeline(
        "translation",
        model=model_name,
        device=selected if selected in {"mps", "cuda"} else -1,
        batch_size=batch_size
    )

def batched(iterable, n: int):
    """Simple batch generator (avoids dependency bloat)"""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def translate_with_progress(
    translator: pipeline,
    texts: List[str],
    show_progress: bool = True,
    desc: str = "Translating"
) -> Iterator[Dict[str, Any]]:
    """
    Translate texts and yield each result immediately with live progress + printing.
    """
    total = len(texts)
    log.info(f"Starting translation of {total} Japanese sentence(s)")

    iterator: Iterator[List[str]] = batched(texts, translator._batch_size)
    
    if total <= translator._batch_size * 2:
        show_progress = False

    pbar = tqdm(
        iterator,
        total=(total + translator._batch_size - 1) // translator._batch_size,
        desc=desc,
        disable=not show_progress,
        leave=False
    )

    translated_count = 0
    for batch in pbar:
        try:
            batch_results = translator(batch)
            for result in batch_results:
                translated_count += 1
                pbar.set_postfix({"done": translated_count, "total": total})
                yield result
        except Exception as e:
            log.error(f"Translation failed on batch: {e}")
            raise

    log.info(f"Translation completed: {translated_count} sentence(s) processed")

# Example usage
if __name__ == "__main__":
    translator = create_translator(batch_size=64)  # Good default for M1/M2

    sample_texts = [
        # "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス",
        "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う二つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア",
        # "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探る",
        # "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオプロジェクトによる",
        # "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリクスを発動",
        # "各国が水面下で熾烈な情報戦を繰り広げる時代に睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリクスを発動作戦を担う数多くの戦争に戦い",
        # "各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリクスを発動作戦を担うスゴーデエージェント黄昏",    
    ]

    translations = translate_with_progress(translator, sample_texts, desc="JA → EN")

    # FIXED: Iterate with zip + enumerate or use itertools.islice for streaming preview
    printed = 0
    max_print = 10
    for ja, res in zip(sample_texts, translations):
        if printed < max_print:
            print(f"JA: {ja}")
            print(f"EN: {res['translation_text']}\n")
            printed += 1
        else:
            # Continue consuming the generator (important!)
            continue

    # If you want to stop early and discard rest (optional, for demo only):
    # from itertools import islice
    # for ja, res in zip(sample_texts, islice(translations, 10)):
    #     print(f"JA: {ja}")
    #     print(f"EN: {res['translation_text']}\n")