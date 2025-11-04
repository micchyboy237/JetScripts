# JetScripts/libs/stanza/common/run_data_objects.py
"""
Run examples demonstrating stanza data object property extensions.
Results are saved to JSON files in the same directory.
"""
import logging
import os
import shutil
from typing import Any, Dict


from jet.file.utils import save_file

from jet.libs.stanza.common.extract_data_objects import extract_backpointer, extract_getter, extract_readonly, extract_setter_getter
from jet.libs.bertopic.examples.mock import load_sample_data
from tqdm import tqdm

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# --------------------------------------------------------------------------- #
# Logging & Progress
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Example functions (return typed values)
# --------------------------------------------------------------------------- #

def example_readonly(text: str) -> Dict[str, Any]:
    """Demonstrate a read-only document property."""
    return extract_readonly(
        input_text=text,
        property_name="some_property",
        property_value=123,
    )

def example_getter(text: str) -> Dict[str, Any]:
    """Show a derived word property combining UPOS+XPOS."""
    return extract_getter(
        input_text=text,
    )

def example_setter_getter(text: str) -> Dict[str, Any]:
    """Illustrate a sentence property with custom setter/getter."""
    return extract_setter_getter(
        input_text=text,
        prop_name="classname",
        set_good_value="good",
        set_bad_internal=2,
    )

def example_backpointer(text: str) -> Dict[str, Any]:
    """Verify back-pointers from words/tokens/entities to their sentence."""
    return extract_backpointer(
        input_text=text,
    )

# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
def main() -> None:
    model = "embeddinggemma"
    chunks = load_sample_data(model=model)

    examples = [
        example_readonly,
        example_getter,
        example_setter_getter,
        example_backpointer,
    ]
    total = len(examples)
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")): 
        sub_output_dir = f"{OUTPUT_DIR}/chunk_{chunk_idx + 1}"
        save_file(chunk, f"{sub_output_dir}/chunk.txt")

        for idx, func in enumerate(examples, start=1):
            logger.info("Running chunk %d - %s (%d/%d)...", chunk_idx + 1, func.__name__, idx, total)
            result = func(chunk)
            result_path = f"{sub_output_dir}/{func.__name__}.json"
            save_file(result, result_path)

        logger.info("All %d examples completed.", total)

if __name__ == "__main__":
    main()