from datetime import datetime, timedelta
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
import psycopg2
from psycopg2 import sql
import scrapy
from jet.scrapers.browser.scrapy.utils import normalize_url
from tqdm import tqdm
import random
import time
import re


DB_CONFIG = {
    "dbname": "anime_db1",
    "user": "jethroestrada",
    "password": "",
    "host": "jetairm1",
    "port": "5432"
}


class AnimeDetails(TypedDict):
    id: int
    url: str
    start_date: str
    end_date: Optional[str]
    status: str
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
    next_episode: Optional[int]
    next_date: Optional[str]


class AnilistHistoryDetailsSpider(scrapy.Spider):
    name = "anilist_history_details_spider"
    table_name = "history"

    def ensure_columns_exist(self):
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch existing columns for the table
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
        """, (self.table_name,))
        existing_columns = {row[0] for row in cursor.fetchall()}

        # Define expected columns
        expected_columns = {
            "start_date": "TEXT",
            "end_date": "TEXT",
            "status": "TEXT",
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
            "tags": "TEXT",
            "next_episode": "INTEGER",
            "next_date": "TEXT",
        }

        # Add missing columns dynamically
        for column, col_type in expected_columns.items():
            if column not in existing_columns:
                logger.info(f"Adding missing column: {column} ({col_type})")
                cursor.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN {} {};").format(
                        sql.Identifier(self.table_name),
                        sql.Identifier(column),
                        sql.SQL(col_type)
                    )
                )

        conn.commit()
        cursor.close()
        conn.close()

    def start_requests(self):
        self.ensure_columns_exist()

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch anime URLs that need updates
        cursor.execute(f"""
            SELECT id, url FROM {self.table_name}
            WHERE status is null
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
            status=self.extract_status(response),
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
            tags=", ".join(self.extract_tags(response)),
            next_episode=self.extract_next_episode(
                response),  # Extracted next episode
            next_date=self.extract_next_date(
                response)         # Extracted next date
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

    def extract_status(self, response) -> Optional[str]:
        return response.css("div.data-set").xpath(
            "div[contains(text(), 'Status')]/following-sibling::div/text()"
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

    def extract_next_episode(self, response) -> Optional[int]:
        # Example: Extract the episode number, e.g., "Ep 25"
        episode_text = response.css(
            ".airing-countdown .countdown span::text").get()
        if episode_text:
            match = re.search(r"Ep (\d+)", episode_text)
            return int(match.group(1)) if match else None
        return None

    def extract_next_date(self, response) -> Optional[str]:
        # Example: Extract the next airing time in the format "10h 13m"
        date_text = response.css(
            ".airing-countdown .countdown span::text").getall()

        if date_text:
            # Join the parts of the time (e.g., '10h', '13m')
            time_str = "".join(date_text).strip()

            # Regex pattern to extract optional days, hours, and minutes
            match = re.search(
                r'(\d+)d\s*(\d+)h\s*(\d+)m|(\d+)h\s*(\d+)m', time_str)

            if match:
                # Extract days, hours, and minutes with optional days
                if match.group(1) and match.group(2) and match.group(3):
                    days = int(match.group(1))
                    hours = int(match.group(2))
                    minutes = int(match.group(3))
                else:
                    days = 0  # Default to 0 if no days are provided
                    hours = int(match.group(4))  # Extract the hours
                    minutes = int(match.group(5))  # Extract the minutes

                # Get current time and add the remaining time
                current_time = datetime.now()
                next_time = current_time + \
                    timedelta(days=days, hours=hours, minutes=minutes)

                # Set the seconds to 00 and format the date as "Month Day, Year Hour:Minute:00"
                next_time = next_time.replace(second=0)

                return next_time.strftime("%B %d, %Y %H:%M:%S")

        return None

    def extract_tags(self, response) -> List[str]:
        return [tag.strip() for tag in response.css("div.tags div.tag a::text").getall()]

    def save_to_db(self, anime_details: AnimeDetails):
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(f"""
        UPDATE {self.table_name}
        SET synopsis=COALESCE(%s, synopsis),
            start_date=COALESCE(%s, start_date),
            end_date=COALESCE(%s, end_date), 
            status=COALESCE(%s, status), 
            members=COALESCE(%s, members),
            popularity=COALESCE(%s, popularity),
            demographic=COALESCE(%s, demographic),
            average_score=COALESCE(%s, average_score),
            mean_score=COALESCE(%s, mean_score),
            favorites=COALESCE(%s, favorites),
            producers=COALESCE(%s, producers),
            source=COALESCE(%s, source),
            japanese=COALESCE(%s, japanese),
            english=COALESCE(%s, english),
            synonyms=COALESCE(%s, synonyms),
            tags=COALESCE(%s, tags),
            next_episode=COALESCE(%s, next_episode),
            next_date=%s  -- Directly update next_date even if it has an existing value
        WHERE id=%s
        """, (
            anime_details["synopsis"],
            anime_details["start_date"],
            anime_details["end_date"],
            anime_details["status"],
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
            anime_details["next_episode"],    # New field
            anime_details["next_date"],       # New field
            anime_details["id"]
        ))

        conn.commit()
        conn.close()

    def closed(self, reason):
        logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings)
    process.crawl(AnilistHistoryDetailsSpider)
    process.start()
