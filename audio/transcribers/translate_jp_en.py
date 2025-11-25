import os
import shutil
import logging
import torch
import gc
from typing import Iterator, Dict, Any, Optional
from transformers import pipeline
from tqdm.auto import tqdm
from pathlib import Path
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def create_translator(
    model_name: str = "Helsinki-NLP/opus-mt-ja-en",
    device: str = "auto",
    batch_size: int = 32,        # Reduced from 64 → safer on M1
    use_fp16: bool = True,       # MPS supports bfloat16/FP16
) -> pipeline:
    """
    Create memory-efficient translation pipeline.
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

    log.info(f"Loading model '{model_name}' on {selected.upper()} (batch_size={batch_size})")

    # Critical: inference mode + no_grad for minimal memory
    torch_device = 0 if selected in {"cuda", "mps"} else -1

    translator = pipeline(
        "translation",
        model=model_name,
        device=torch_device,
        batch_size=batch_size,
        torch_dtype=torch.float16 if use_fp16 and selected in {"cuda", "mps"} else torch.float32,
    )

    return translator


def batched(iterable: Iterator[str], n: int) -> Iterator[list[str]]:
    """Lightweight batch generator that consumes iterator lazily."""
    batch = []
    for item in iterable:
        batch.append(item.strip())
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def text_file_iterator(file_path: Path) -> Iterator[str]:
    """Stream lines from file – zero memory overhead for huge files."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


@torch.inference_mode()
def translate_streaming(
    translator: pipeline,
    texts_iter: Iterator[str],
    desc: str = "Translating JA → EN",
    max_sentences: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Fully streaming translation – never holds more than one batch in memory.
    """
    total_processed = 0
    batch_size = translator._batch_size or 1

    # Wrap iterator with optional limit
    if max_sentences:
        texts_iter = (x for _, x in zip(range(max_sentences), texts_iter))

    batch_gen = batched(texts_iter, batch_size)
    pbar = tqdm(batch_gen, desc=desc, unit="batch", leave=False)

    for batch in pbar:
        try:
            results = translator(batch)  # List[Dict]
            for res in results:
                yield res
                total_processed += 1
                pbar.set_postfix({"sentences": total_processed})

            # Force cleanup after each batch
            del batch, results
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            log.error(f"Batch failed: {e}")
            raise

    log.info(f"Translation finished – {total_processed} sentences processed")


# ———————— Example Usage (Streaming from file) ————————
if __name__ == "__main__":
    # Recommended safe settings for M1 Mac
    translator = create_translator(
        model_name="Helsinki-NLP/opus-mt-ja-en",
        batch_size=24,      # 24–32 is sweet spot on M1/M2 16GB
        use_fp16=True
    )

    input_file = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/transcribers/generated/run_jp_transcriber/translations/translation.txt")  # One sentence per line

    if not input_file.exists():
        log.error(f"File {input_file} not found!")
        exit(1)

    # Fully streaming – memory stays < 6 GB even with millions of lines
    translations = translate_streaming(
        translator,
        text_file_iterator(input_file),
        desc="JA → EN",
        max_sentences=None  # Set to e.g. 10_000 for testing
    )

    for result in translations:
        save_file(result, f"{OUTPUT_DIR}/translations.jsonl", append=True, verbose=False)
        print(f"EN: {result['translation_text']}\n")
