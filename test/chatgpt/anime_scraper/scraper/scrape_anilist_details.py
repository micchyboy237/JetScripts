from typing import List, TypedDict, Optional
from jet.utils.commands import copy_to_clipboard
from jet.transformers.formatters import format_json
from jet.logger import logger
import scrapy.http
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
import re


DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


class AnimeDetails(TypedDict):
    id: int
    url: str
    start_date: str
    end_date: Optional[str]
    members: Optional[int]
    synopsis: str
    popularity: int
    demographic: str
    average_score: Optional[int]
    mean_score: Optional[int]
    favorites: Optional[int]
    producers: Optional[str]
    source: Optional[str]
    japanese: Optional[str]
    english: Optional[str]
    synonyms: Optional[str]
    tags: Optional[str]


class AnilistDetailsSpider(scrapy.Spider):
    name = "anilistdetails_spider"
    table_name = "trending"

    def start_requests(self):
        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

        # Fetch existing columns
        cursor.execute(f"PRAGMA table_info({self.table_name});")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Define expected columns
        expected_columns = {
            "start_date": "TEXT",
            "end_date": "TEXT",
            "members": "INTEGER",
            "synopsis": "TEXT",
            "popularity": "INTEGER",
            "demographic": "TEXT",
            "average_score": "INTEGER",
            "mean_score": "INTEGER",
            "favorites": "INTEGER",
            "producers": "TEXT",
            "source": "TEXT",
            "japanese": "TEXT",
            "english": "TEXT",
            "synonyms": "TEXT",
            "tags": "TEXT"
        }

        # Add missing columns dynamically
        for column, col_type in expected_columns.items():
            if column not in existing_columns:
                logger.info(f"Adding missing column: {column} ({col_type})")
                cursor.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN {column} {col_type};")

        conn.commit()

        # Fetch anime URLs that need updates
        cursor.execute(f"""
            SELECT id, url FROM {self.table_name}
            WHERE end_date IS NULL
            OR members IS NULL
            OR synopsis IS NULL
            OR popularity IS NULL
            OR demographic IS NULL
            OR average_score IS NULL
            OR mean_score IS NULL
            OR favorites IS NULL
            OR producers IS NULL
            OR source IS NULL
            OR japanese IS NULL
            OR english IS NULL
            OR synonyms IS NULL
            OR tags IS NULL
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
                wait_until=EC.presence_of_element_located(
                    (By.TAG_NAME, ".content .description")),
                meta={"anime_id": anime_id}
            )

    def parse(self, response: scrapy.http.Response):
        anime_id = response.meta["anime_id"]

        anime_details = AnimeDetails(
            id=anime_id,
            url=quote(response.url, safe=":/"),
            synopsis=self.extract_synopsis(response),
            start_date=self.extract_start_date(response),
            end_date=self.extract_end_date(response),
            members=self.extract_members(response),
            popularity=self.extract_popularity(response),
            demographic=self.extract_demographic(response),
            average_score=self.extract_average_score(response),
            mean_score=self.extract_mean_score(response),
            favorites=self.extract_favorites(response),
            producers=", ".join(self.extract_producers(response)),
            source=self.extract_source(response),
            japanese=self.extract_japanese(response),
            english=self.extract_english(response),
            synonyms=", ".join(self.extract_synonyms(response)),
            tags=", ".join(self.extract_tags(response))
        )

        self.save_to_db(anime_details)

        yield anime_details

    def extract_synopsis(self, response) -> Optional[str]:
        synopsis_parts = response.css(
            "p.description, p.description.content-wrap"
        ).xpath(".//text()").getall()

        # Remove leading/trailing spaces and filter out empty strings
        synopsis_parts = [part.strip()
                          for part in synopsis_parts if part.strip()]

        # Use set to remove duplicates while preserving order
        seen = set()
        unique_synopsis = [part for part in synopsis_parts if not (
            part in seen or seen.add(part))]

        synopsis = " ".join(unique_synopsis)

        # Fix spaces before punctuation
        synopsis = re.sub(r"\s+([.,!?])", r"\1", synopsis)

        return synopsis if synopsis else None

    def extract_members(self, response) -> Optional[int]:
        return 0

    def extract_demographic(self, response) -> Optional[str]:
        return ""

    def extract_start_date(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Start Date')]/following-sibling::div/text()"
        ).get()

    def extract_end_date(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'End Date')]/following-sibling::div/text()"
        ).get()

    def extract_average_score(self, response) -> Optional[int]:
        score_text = response.css("div.data-set").xpath(
            "div[contains(text(), 'Average Score')]/following-sibling::div/text()"
        ).get()
        return int(score_text.replace('%', '').strip()) if score_text else None

    def extract_mean_score(self, response) -> Optional[int]:
        score_text = response.css("div.data-set").xpath(
            "div[contains(text(), 'Mean Score')]/following-sibling::div/text()"
        ).get()
        return int(score_text.replace('%', '').strip()) if score_text else None

    def extract_popularity(self, response) -> Optional[int]:
        pop_text = response.css("div.data-set").xpath(
            "div[contains(text(), 'Popularity')]/following-sibling::div/text()"
        ).get()
        return int(pop_text.strip()) if pop_text else None

    def extract_favorites(self, response) -> Optional[int]:
        fav_text = response.css("div.data-set").xpath(
            "div[contains(text(), 'Favorites')]/following-sibling::div/text()"
        ).get()
        return int(fav_text.strip()) if fav_text else None

    def extract_genres(self, response) -> List[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Genres')]/following-sibling::div//a/text()"
        ).getall()

    def extract_studios(self, response) -> List[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Studios')]/following-sibling::div//a/text()"
        ).getall()

    def extract_producers(self, response) -> List[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Producers')]/following-sibling::div//a/text()"
        ).getall()

    def extract_source(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Source')]/following-sibling::div/text()"
        ).get()

    def extract_japanese(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Romaji')]/following-sibling::div/text()"
        ).get()

    def extract_english(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'English')]/following-sibling::div/text()"
        ).get()

    def extract_synonyms(self, response) -> List[str]:
        return [syn.strip() for syn in response.xpath(
            "//div[contains(@class, 'data-set')][div[contains(text(), 'Synonyms')]]/div[contains(@class, 'value')]/span/text()"
        ).getall()]

    def extract_tags(self, response) -> List[str]:
        return [tag.strip() for tag in response.css("div.tags div.tag a::text").getall()]

    def save_to_db(self, anime_details: AnimeDetails):
        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

        cursor.execute(f"""
        UPDATE {self.table_name}
        SET synopsis=COALESCE(?, synopsis),
            start_date=COALESCE(?, start_date),
            end_date=COALESCE(?, end_date), 
            members=COALESCE(?, members),
            popularity=COALESCE(?, popularity),
            demographic=COALESCE(?, demographic),
            average_score=COALESCE(?, average_score),
            mean_score=COALESCE(?, mean_score),
            favorites=COALESCE(?, favorites),
            producers=COALESCE(?, producers),
            source=COALESCE(?, source),
            japanese=COALESCE(?, japanese),
            english=COALESCE(?, english),
            synonyms=COALESCE(?, synonyms),
            tags=COALESCE(?, tags)
        WHERE id=?
        """, (
            anime_details["synopsis"],
            anime_details["start_date"],
            anime_details["end_date"],
            anime_details["members"],
            anime_details["popularity"],
            anime_details["demographic"],
            anime_details["average_score"],
            anime_details["mean_score"],
            anime_details["favorites"],
            anime_details["producers"],
            anime_details["source"],
            anime_details["japanese"],
            anime_details["english"],
            anime_details["synonyms"],
            anime_details["tags"],
            anime_details["id"]
        ))

        conn.commit()
        conn.close()

    def closed(self, reason):
        logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings)
    process.crawl(AnilistDetailsSpider)
    process.start()
