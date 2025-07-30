import json
import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
from jet.wordnet.keywords.spelling_corrector import SpellingCorrector

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    misspelled_texts = [
        "Helo, wrld! I am fien.",
        "Shee is he or shee."
    ]
    corrected_texts = [
        "Hello, world! I am fine.",
        "She is he or she."
    ]

    text_corrector = SpellingCorrector(base_sentences=corrected_texts)
    results_stream = text_corrector.autocorrect_texts(misspelled_texts)

    autocorrect_results = []
    for result in results_stream:
        logger.success(f"Result: {json.dumps(result, indent=2)}")
        autocorrect_results.append(result)

    save_file({
        "misspelled_texts": misspelled_texts,
        "count": len(autocorrect_results),
        "autocorrect_results": autocorrect_results
    }, f"{OUTPUT_DIR}/autocorrect_results.json")

    save_file({
        "misspelled_texts": misspelled_texts,
        "count": len(text_corrector.unknown_words),
        "unknown_words": text_corrector.unknown_words
    }, f"{OUTPUT_DIR}/unknown_words.json")
