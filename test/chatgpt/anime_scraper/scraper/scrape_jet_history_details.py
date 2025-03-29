from typing import List, TypedDict, Optional
from jet.utils.commands import copy_to_clipboard
from jet.transformers.formatters import format_json
from jet.logger import logger
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from jet.scrapers.browser.scrapy import settings, SeleniumRequest
from urllib.parse import quote
import sqlite3
import scrapy
from jet.scrapers.browser.scrapy.utils import normalize_url
from tqdm import tqdm
import random
import time


DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


class AnimeDetails(TypedDict):
    url: str
    synopsis: str
    genres: List[str]
    popularity: int
    anime_type: str
    demographic: str


class JetAnimeHistoryDetailsSpider(scrapy.Spider):
    name = "jetanimehistorydetails_spider"
    table_name = "jet_history"

    def start_requests(self):
        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

        # Fetch existing columns
        cursor.execute(f"PRAGMA table_info({self.table_name});")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Define expected columns
        expected_columns = {
            "synopsis": "TEXT",
            "genres": "TEXT",
            "popularity": "INTEGER",
            "anime_type": "TEXT",
            "demographic": "TEXT"
        }

        # Add missing columns dynamically
        for column, col_type in expected_columns.items():
            if column not in existing_columns:
                logger.info(f"Adding missing column: {column} ({col_type})")
                cursor.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN {column} {col_type};")

        conn.commit()  # Save changes

        # Now the table is guaranteed to have the required columns
        cursor.execute(f"""
            SELECT id, url FROM {self.table_name}
            WHERE end_date IS NULL
            OR members IS NULL
            OR synopsis IS NULL
            OR popularity IS NULL
            OR demographic IS NULL
        """)

        anime_data = [(row[0], quote(row[1], safe=":/"))
                      for row in cursor.fetchall()]
        conn.close()

        for anime_id, url in tqdm(anime_data, desc="Scraping details..."):
            delay = random.uniform(2, 5)
            logger.info(
                f"Waiting for {delay:.2f} seconds before next request...")
            time.sleep(delay)

            yield SeleniumRequest(
                url=url,
                callback=self.parse,
                wait_time=3,
                wait_until=EC.presence_of_element_located((By.TAG_NAME, "h2")),
                meta={"anime_id": anime_id}  # Pass id to the parse method
            )

    def parse(self, response):
        driver: WebDriver = response.request.meta['driver']

        anime_id = response.meta["anime_id"]  # Retrieve id

        anime_details = AnimeDetails(
            id=anime_id,  # Include id
            url=quote(response.url, safe=":/"),
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

        # # Ensure new columns exist, ignoring errors if they already exist
        # for column, col_type in [
        #     ("synopsis", "TEXT"),
        #     ("genres", "TEXT"),
        #     ("popularity", "INTEGER"),
        #     ("anime_type", "TEXT"),
        #     ("demographic", "TEXT")
        # ]:
        #     try:
        #         cursor.execute(
        #             f"ALTER TABLE anime ADD COLUMN {column} {col_type};")
        #     except sqlite3.OperationalError:
        #         pass  # Column already exists

        cursor.execute(f"PRAGMA table_info({self.table_name});")
        existing_columns = {row[1] for row in cursor.fetchall()}

        for column, col_type in [("synopsis", "TEXT"), ("genres", "TEXT"),
                                 ("popularity", "INTEGER"), ("anime_type", "TEXT"),
                                 ("demographic", "TEXT")]:
            if column not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN {column} {col_type};")

        # Update the existing row with new data
        cursor.execute(f"""
        UPDATE {self.table_name}
        SET synopsis=COALESCE(?, synopsis), 
            genres=COALESCE(?, genres), 
            popularity=COALESCE(?, popularity), 
            anime_type=COALESCE(?, anime_type), 
            demographic=COALESCE(?, demographic)
        WHERE url=?
        """, (
            anime_details['synopsis'],
            ", ".join(anime_details['genres']),
            anime_details['popularity'],
            anime_details['anime_type'],
            anime_details['demographic'],
            normalize_url(anime_details['url'])
        ))

        conn.commit()
        conn.close()

    def closed(self, reason):
        logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings)
    process.crawl(JetAnimeHistoryDetailsSpider)
    process.start()
