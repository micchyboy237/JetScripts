import json
import os
from jet.file.utils import save_file
from jet.logger import logger
from jet.wordnet.keywords.spelling_corrector import SpellingCorrector

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == '__main__':
    text_corrector = SpellingCorrector()
    english_texts = [
        "Helo, wrld! I am fien.",
        "Shee is he or shee."
    ]
    texts = english_texts

    results_stream = text_corrector.autocorrect_texts(texts)

    results = []
    for result in results_stream:
        logger.success(f"Result: {json.dumps(result, indent=2)}")
        results.append(result)

    save_file({
        "texts": texts,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
