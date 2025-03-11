import sqlite3
import os
import json
import re

from urllib.parse import urlparse
from jet.wordnet.n_grams import group_sentences_by_ngram
from jet.wordnet.similarity import get_similar_texts


def extract_unique_anime_titles(urls):
    anime_titles = set()

    # Regex pattern to extract anime titles from URLs
    pattern = re.compile(
        r"https://aniwatchtv.to/(?:watch/)?([a-z0-9-]+)(?:-\d+)?(?:\?|$)")

    for url in urls:
        parsed_url = urlparse(url)

        # Skip search and home URLs
        if "/search" in parsed_url.path or parsed_url.path in ["/", "/home"]:
            continue

        match = pattern.search(url)
        if match:
            title = match.group(1)
            # Remove last part if it's a number
            title = re.sub(r"-\d+$", "", title)
            title = title.replace("-", " ").title()
            anime_titles.add(title)

    return sorted(anime_titles)


# Example usage
if __name__ == "__main__":
    # Path to Opera history file
    db_path = os.path.expanduser(
        "~/Library/Application Support/com.operasoftware.Opera/Default/History")

    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
    else:
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Query to get all URLs
            query = "SELECT url FROM urls"
            cursor.execute(query)

            # Fetch results
            results = cursor.fetchall()

            # Filter results based on domain containing "aniwatch"
            filtered_urls = [row[0]
                             for row in results if "aniwatch" in urlparse(row[0]).netloc]

            # Extract unique anime titles
            unique_titles = extract_unique_anime_titles(filtered_urls)
            print(unique_titles)

            # Save results as JSON
            output_file = os.path.expanduser("data/aniwatch_history.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(unique_titles, file, indent=4)

            # Close connection
            conn.close()

            print(f"Extracted URLs saved to {output_file}")

        except sqlite3.OperationalError as e:
            print(f"SQLite operational error: {e}")
