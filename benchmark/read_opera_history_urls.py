import sqlite3
import os
import json
import re
from urllib.parse import urlparse


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
    db_path = os.path.expanduser(
        "~/Library/Application Support/com.operasoftware.Opera/Default/History")
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data"

    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
    else:
        conn = None
        try:
            # Connect with timeout and WAL mode
            # Allow waiting for 10 sec
            conn = sqlite3.connect(db_path, timeout=10)
            # Set write-ahead logging mode
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()

            query = "SELECT url FROM urls"
            cursor.execute(query)

            results = cursor.fetchall()

            # Filter results based on domain containing "aniwatch"
            filtered_urls = [row[0]
                             for row in results if "aniwatch" in urlparse(row[0]).netloc]

            unique_titles = extract_unique_anime_titles(filtered_urls)
            print(unique_titles)

            # Save results as JSON
            output_file = f"{output_dir}/aniwatch_history.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(unique_titles, file, indent=4)

            print(f"Extracted URLs saved to {output_file}")

        except sqlite3.OperationalError as e:
            print(f"SQLite operational error: {e}")

        finally:
            if conn:
                conn.close()  # Ensure connection is always closed
