import scrapy
import sqlite3
from jet.scrapers.browser.scrapy import settings, SeleniumRequest
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from typing import List, TypedDict, Optional

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


class AnimeDetails(TypedDict):
    url: str
    synopsis: str
    genres: List[str]
    popularity: int
    anime_type: str
    demographic: str


class MyAnimDetailsSpider(scrapy.Spider):
    name = "myanimdetails_spider"

    def start_requests(self):
        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

        # Select only rows where any of the target columns are NULL
        cursor.execute("""
            SELECT url FROM anime
            WHERE synopsis IS NULL 
            OR genres IS NULL 
            OR popularity IS NULL 
            OR anime_type IS NULL 
            OR demographic IS NULL
        """)

        urls = [row[0] for row in cursor.fetchall()]
        conn.close()

        for url in urls:
            yield SeleniumRequest(
                url=url,
                callback=self.parse,
                wait_time=3,
                wait_until=EC.presence_of_element_located((By.TAG_NAME, "h2")),
            )

    def parse(self, response):
        driver: WebDriver = response.request.meta['driver']

        anime_details = AnimeDetails(
            url=response.url,
            synopsis=self.extract_synopsis(response),
            genres=self.extract_genres(response),
            popularity=self.extract_popularity(response),
            anime_type=self.extract_type(response),
            demographic=self.extract_demographic(response),
        )

        self.save_to_db(anime_details)

        yield anime_details

    def extract_text(self, response, header_text: str) -> Optional[str]:
        """Extracts text data based on header."""
        element = response.xpath(
            f"//h2[contains(text(), '{header_text}')]/following-sibling::div[1]/text()").get()
        return element.strip() if element else None

    def extract_synopsis(self, response) -> Optional[str]:
        """Extracts synopsis from the anime page."""
        return response.css("p[itemprop='description']::text").get()

    def extract_genres(self, response) -> List[str]:
        """Extracts genres from the page."""
        return response.css("div.spaceit_pad:contains('Genres') a::text").getall()

    def extract_type(self, response) -> List[str]:
        """Extracts type from the page."""
        return response.css("div.spaceit_pad:contains('Type') a::text").get()

    def extract_demographic(self, response) -> List[str]:
        """Extracts demographic from the page."""
        return response.css("div.spaceit_pad:contains('Demographic') a::text").get()

    def extract_popularity(self, response) -> Optional[int]:
        """Extracts popularity ranking from the page."""
        pop_text = response.xpath(
            "//span[contains(text(), 'Popularity')]/following-sibling::text()").get()
        return int(pop_text.replace("#", "").strip()) if pop_text else None

    def save_to_db(self, anime_details: AnimeDetails):
        """Saves the extracted data to the database."""
        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

        # Ensure new columns exist, ignoring errors if they already exist
        for column, col_type in [
            ("synopsis", "TEXT"),
            ("genres", "TEXT"),
            ("popularity", "INTEGER"),
            ("anime_type", "TEXT"),
            ("demographic", "TEXT")
        ]:
            try:
                cursor.execute(
                    f"ALTER TABLE anime ADD COLUMN {column} {col_type};")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Update the existing row with new data
        cursor.execute("""
        UPDATE anime
        SET synopsis=?, genres=?, popularity=?, anime_type=?, demographic=?
        WHERE url=?
        """, (
            anime_details['synopsis'],
            ", ".join(anime_details['genres']),
            anime_details['popularity'],
            anime_details['anime_type'],
            anime_details['demographic'],
            anime_details['url']
        ))

        conn.commit()
        conn.close()

    def closed(self, reason):
        logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings)
    process.crawl(MyAnimDetailsSpider)
    process.start()
