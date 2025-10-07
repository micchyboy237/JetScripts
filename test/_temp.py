from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from jet.adapters.bertopic import BERTopic
from jet.wordnet.text_chunker import chunk_texts_fast
from jet.file.utils import save_file
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

EMBED_MODEL = "embeddinggemma"

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data[:100]  # Limit to 10 documents for example
documents = chunk_texts_fast(
    documents,
    chunk_size=128,
    chunk_overlap=32,
    model=EMBED_MODEL,
)
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")

topic_model = BERTopic(vectorizer_model=vectorizer)
topics, probs = topic_model.fit_transform(documents)

topic_info = topic_model.get_topic_info()
save_file(topic_info, f"{OUTPUT_DIR}/results/topic_info.json")
