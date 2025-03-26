from typing import Optional, TypedDict
from retriever.search import hybrid_search
import sqlite3

DB_PATH = "data/top_upcoming_anime.db"


class ScrapedData(TypedDict):
    rank: int
    title: str
    url: str
    image_url: str
    score: float
    episodes: int
    start_date: str
    end_date: Optional[str]
    status: str
    members: int
    synopsis: str
    genres: list[str]
    popularity: int
    anime_type: str
    demographic: str


def fetch_anime_by_ids(anime_ids):
    """Fetch anime details from the database based on retrieved indices."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(anime_ids))
    query = f"""
        SELECT rank, title, url, image_url, score, episodes, start_date, 
               end_date, status, members, synopsis, genres, popularity, 
               anime_type, demographic 
        FROM anime 
        WHERE rank IN ({placeholders})
    """

    cursor.execute(query, anime_ids)
    results = cursor.fetchall()
    conn.close()

    return [ScrapedData(*row) for row in results]


def retrieve_anime_info(query):
    anime_ids = hybrid_search(query)

    if not anime_ids:
        return "No relevant anime found."

    anime_list = fetch_anime_by_ids(anime_ids)

    response = "\n\n".join([
        f"🔹 **{anime['title']}**\n"
        f"📖 Synopsis: {anime['synopsis'][:200]}...\n"
        f"📺 Episodes: {anime['episodes']} | 🎭 Type: {anime['anime_type']}\n"
        f"📊 Score: {anime['score']} | ⭐ Popularity: {anime['popularity']}\n"
        f"📅 Aired: {anime['start_date']} - {anime['end_date'] or 'Ongoing'}\n"
        f"🔗 More: {anime['url']}"
        for anime in anime_list
    ])

    return response


if __name__ == "__main__":
    while True:
        query = input("Enter your anime query: ")
        print("\n🔹 AI Response:\n", retrieve_anime_info(query))
