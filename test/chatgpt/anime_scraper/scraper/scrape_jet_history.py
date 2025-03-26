import random
import re
import sqlite3
import json
from urllib.parse import urlencode
from typing import List, TypedDict, Optional

from jet.logger.timer import sleep_countdown
from tqdm import tqdm
import scrapy
from jet.scrapers.browser.scrapy import settings, SeleniumRequest, normalize_url
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


ANIME_TITLES_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data/aniwatch_history.json"
DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]
    title: str
    url: str
    image_url: str
    score: Optional[float]
    episodes: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    status: str
    members: Optional[int]
    anime_type: Optional[str]


class JetAnimeHistorySpider(scrapy.Spider):
    name = "jetanimehistory_spider"
    table_name = "jet_history"

    def start_requests(self):
        # Load anime titles from the JSON file
        with open(ANIME_TITLES_PATH, "r", encoding="utf-8") as file:
            anime_titles = json.load(file)

        for title in tqdm(anime_titles, desc="Scraping list..."):
            # Delay to prevent spam
            delay = random.uniform(2, 5)
            sleep_countdown(delay, f"Delaying:")

            # Modify title if it's shorter than 3 characters
            if len(title) < 3:
                title = f"{title} {title}"

            # Construct search URL
            query_params = {"q": title}
            search_url = f"https://myanimelist.net/search/all?{urlencode(query_params)}"

            yield SeleniumRequest(
                url=search_url,
                callback=self.parse,
                wait_time=3,
                wait_until=EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "article div.list")
                ),
                screenshot=True,
                meta={"original_title": title},
            )

    def parse(self, response):
        driver: WebDriver = response.request.meta["driver"]
        original_title = response.request.meta["original_title"]

        with sqlite3.connect(f"{DATA_DIR}/anime.db", timeout=10) as conn:
            cursor = conn.cursor()

            # Enable WAL mode for concurrent access
            cursor.execute("PRAGMA journal_mode=WAL;")

            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                rank INTEGER,
                title TEXT,
                url TEXT,
                image_url TEXT,
                score REAL,
                episodes INTEGER,
                start_date TEXT,
                end_date TEXT,
                status TEXT,
                members INTEGER,
                anime_type TEXT
            )
            """)

            results = []

            for anime in response.css("article div.list"):
                title_element = anime.css(".title a")
                title = title_element.css("::text").get()

                if "href" not in title_element.attrib:
                    logger.warning(f"Missing href for: {title}")
                    continue

                url = title_element.attrib["href"]
                anime_id = self.extract_anime_id(url)

                image_url = anime.css(
                    ".picSurround a img::attr(data-src)").get()

                details = anime.css(".information").get()

                anime_type = re.search(
                    r'>(TV|Movie|OVA|ONA|Special|Music)<', details)
                anime_type = anime_type.group(1) if anime_type else None

                episodes_match = re.search(r'(\d+) eps', details)
                episodes = int(episodes_match.group(
                    1)) if episodes_match else None

                score_match = re.search(r'Scored ([\d.]+)', details)
                score = float(score_match.group(1)) if score_match else None

                members_match = re.search(r'([\d,]+) members', details)
                members = int(members_match.group(1).replace(
                    ",", "")) if members_match else None

                anime_data = ScrapedData(
                    id=str(anime_id),
                    rank=None,
                    title=title.strip() if title else original_title,
                    url=url,
                    image_url=image_url if image_url else "",
                    score=score,
                    episodes=episodes,
                    start_date=None,
                    end_date=None,
                    status="Finished" if episodes else "Ongoing",
                    members=members,
                    anime_type=anime_type,
                )

                results.append(anime_data)

            # Batch insert for efficiency
            try:
                cursor.executemany(f"""
                INSERT OR IGNORE INTO {self.table_name} 
                (id, rank, title, url, image_url, score, episodes, start_date, end_date, status, members, anime_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(d["id"], d["rank"], d["title"], d["url"], d["image_url"], d["score"],
                       d["episodes"], d["start_date"], d["end_date"], d["status"], d["members"], d["anime_type"])
                      for d in results])

                conn.commit()
            except sqlite3.OperationalError as e:
                logger.error(f"Database error: {e}")

            for item in results:
                yield item

    def extract_anime_id(self, url: str) -> Optional[str]:
        """Extract anime ID from MyAnimeList URL."""
        match = re.search(r'myanimelist\.net/anime/(\d+)', url)
        return match.group(1) if match else None

    def closed(self, reason):
        copy_to_clipboard(self.results)
        logger.newline()
        logger.debug(f"\nFinal Results ({len(self.results)}):")
        for item in self.results:
            logger.success(format_json(item))
        logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings)
    process.crawl(JetAnimeHistorySpider)
    process.start()
