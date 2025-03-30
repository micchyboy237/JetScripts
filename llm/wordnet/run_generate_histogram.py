import os
from jet.utils.text import remove_non_alphanumeric
from jet.wordnet.n_grams import count_ngrams
from jet.wordnet.words import count_words
import psycopg
from psycopg.rows import dict_row
from typing import Optional, TypedDict

from jet.file.utils import load_file, save_file
from jet.wordnet.histogram import TextAnalysis, generate_histogram

DB_CONFIG = {
    "dbname": "anime_db1",
    "user": "jethroestrada",
    "password": "",
    "host": "jetairm1",
    "port": "5432"
}
TABLE_NAME = "history"


def load_all_records():
    conn = psycopg.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        autocommit=False,  # Enable manual transaction control
        row_factory=dict_row
    )

    query = f"SELECT * FROM {TABLE_NAME};"
    with conn.cursor() as cur:
        cur.execute(query)
        results = cur.fetchall()
        return results


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]
    title: Optional[str]
    url: Optional[str]
    image_url: Optional[str]
    score: Optional[float]
    episodes: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    next_date: Optional[str]
    status: Optional[str]
    members: Optional[int]
    anime_type: Optional[str]
    average_score: Optional[int]
    mean_score: Optional[int]
    favorites: Optional[int]
    next_episode: Optional[int]
    popularity: Optional[int]
    demographic: Optional[str]
    studios: Optional[str]
    producers: Optional[str]
    source: Optional[str]
    japanese: Optional[str]
    english: Optional[str]
    synonyms: Optional[str]
    tags: Optional[str]
    synopsis: Optional[str]
    genres: Optional[str]


# Prepare data
db_data: list[ScrapedData] = load_all_records()

if __name__ == "__main__":
    output_dir = f'/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/wordnet/data/generated/anime_histogram'

    data = [{"title": d["title"]} for d in db_data]
    texts = [d["title"] for d in db_data]
    formatted_texts_dict = {remove_non_alphanumeric(
        d["title"]).lower(): d for d in db_data}
    formatted_texts = list(formatted_texts_dict.keys())
    ngram_count = count_ngrams(formatted_texts, from_start=True)
    filtered_results = [
        (ngram, count) for ngram, count in ngram_count.items() if ngram in formatted_texts]
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_results = [{"ngram": ngram, "count": count}
                        for ngram, count in filtered_results]
    # Group results by starting ngram
    grouped_results = {}
    for result in filtered_results:
        if result["ngram"] in [
            ngram for ngram_list in list(grouped_results.values())
            for ngram in ngram_list
        ]:
            continue

        if result["count"] > 1:
            matching_ngrams = [item["ngram"] for item in filtered_results
                               if item["ngram"].startswith(result["ngram"])]
            grouped_results[matching_ngrams[0]] = matching_ngrams

    # Revert to actual ngram values
    final_results = {}
    for key, values in grouped_results.items():
        reverted_key = formatted_texts_dict[key]["title"]
        reverted_values = [formatted_texts_dict[value]["title"]
                           for value in values]
        final_results[reverted_key] = reverted_values

    # max_word_count = max([count_words(text) for text in texts])
    # include_keys = ['title']
    # ta = TextAnalysis(data, include_keys=include_keys)

    # histogram_results = ta.generate_histogram(
    #     is_top=True,
    #     from_start=True,
    #     ngram_ranges=[(1, max_word_count)],
    # )
    # filtered_results = [
    #     d for d in histogram_results[0]["results"] if d["ngram"] in texts]

    save_file(final_results, os.path.join(
        output_dir, 'histogram_results.json'))
