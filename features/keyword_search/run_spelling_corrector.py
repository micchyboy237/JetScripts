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
    text_corrector = SpellingCorrector()
    english_texts = [
        "Helo, wrld! I am fien.",
        "Shee is he or shee."
    ]
    texts = english_texts

    results_stream = text_corrector.autocorrect_texts(texts)

    autocorrect_results = []
    for result in results_stream:
        logger.success(f"Result: {json.dumps(result, indent=2)}")
        autocorrect_results.append(result)

    save_file({
        "texts": texts,
        "count": len(autocorrect_results),
        "autocorrect_results": autocorrect_results
    }, f"{OUTPUT_DIR}/autocorrect_results.json")

    save_file({
        "texts": texts,
        "count": len(text_corrector.unknown_words),
        "unknown_words": text_corrector.unknown_words
    }, f"{OUTPUT_DIR}/unknown_words.json")
