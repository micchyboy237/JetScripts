import os
import shutil

from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.utils import clean_punctuations
from jet.transformers.formatters import format_json
from jet.wordnet.keywords.helpers import extract_query_candidates, preprocess_texts
from jet.wordnet.n_grams import count_ngrams
from jet.wordnet.words import get_words


output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(output_dir, ignore_errors=True)

query = "React web"

candidates = extract_query_candidates(query)
logger.success(format_json(candidates))
save_file(candidates, f"{output_dir}/candidates.json")
