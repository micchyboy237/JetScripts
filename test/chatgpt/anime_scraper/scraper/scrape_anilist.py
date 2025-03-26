import scrapy
import sqlite3


class AniListSpider(scrapy.Spider):
    name = "anilist_scraper"
    start_urls = ["https://anilist.co/search/anime/trending"]

    def parse(self, response):
        conn = sqlite3.connect("data/top_upcoming_anime.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS anime (
            title TEXT, synopsis TEXT, status TEXT, episodes TEXT, airing TEXT, source TEXT
        )
        """)

        for anime in response.css(".media-card"):
            title = anime.css(".title a::text").get()
            synopsis = anime.css(".description::text").get()
            status = anime.css(".airing-status::text").get()
            episodes = anime.css(".episodes::text").get()
            airing = anime.css(".airing-time::text").get()

            if title:
                cursor.execute("INSERT INTO anime VALUES (?, ?, ?, ?, ?, ?)",
                               (title.strip(), synopsis.strip() if synopsis else "Unknown",
                                status.strip() if status else "Unknown",
                                episodes.strip() if episodes else "Unknown",
                                airing.strip() if airing else "Unknown",
                                "AniList"))

        conn.commit()
        conn.close()
