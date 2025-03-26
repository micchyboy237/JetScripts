import sqlite3
import os
import json
import re
from urllib.parse import urlparse


def extract_unique_anime_titles(urls):
    """Extract unique anime titles from a list of URLs."""
    anime_titles = set()

    # Regex pattern to extract anime titles from URLs
    pattern = re.compile(
        r"https://aniwatchtv.to/(?:watch/)?([a-z0-9-]+)(?:-\d+)?(?:\?|$)"
    )

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


def extract_opera_history(db_path):
    """Extract URLs from Opera history SQLite database."""
    if not os.path.exists(db_path):
        print(f"Opera database file not found: {db_path}")
        return []

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()

        query = "SELECT url FROM urls"
        cursor.execute(query)

        results = cursor.fetchall()

        return [row[0] for row in results if "aniwatch" in urlparse(row[0]).netloc]

    except sqlite3.OperationalError as e:
        print(f"SQLite operational error: {e}")
        return []

    finally:
        if conn:
            conn.close()


def extract_chrome_history(json_path):
    """Extract URLs from Chrome history JSON file."""
    if not os.path.exists(json_path):
        print(f"Chrome history file not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return [entry["url"] for entry in data if "aniwatch" in urlparse(entry["url"]).netloc]

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading Chrome history JSON: {e}")
        return []


if __name__ == "__main__":
    # Paths
    opera_db = os.path.expanduser(
        "~/Library/Application Support/com.operasoftware.Opera/Default/History"
    )
    chrome_json = "/Users/jethroestrada/Downloads/history.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data"

    # Extract history from both sources
    opera_urls = extract_opera_history(opera_db)
    chrome_urls = extract_chrome_history(chrome_json)

    # Merge and extract unique anime titles
    all_urls = list(set(opera_urls + chrome_urls))
    unique_titles = extract_unique_anime_titles(all_urls)

    print(unique_titles)

    # Save results as JSON
    output_file = f"{output_dir}/aniwatch_history.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(unique_titles, file, indent=4)

    print(f"Extracted URLs saved to {output_file}")
