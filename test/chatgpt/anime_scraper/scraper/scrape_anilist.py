import datetime
import re
import scrapy
import sqlite3
from urllib.parse import quote, urljoin
from jet.scrapers.browser.scrapy import settings, SeleniumRequest, normalize_url
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from jet.logger import logger, sleep_countdown
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from typing import List, TypedDict, Optional

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]  # Rank is not available in the given HTML
    title: str
    url: str
    image_url: str
    score: Optional[float]  # Converted from percentage
    episodes: Optional[int]
    start_date: str
    end_date: Optional[str]  # No end_date found in the given HTML
    next_episode: Optional[int]  # Extracted episode number
    next_date: Optional[str]  # Converted to "MMMM D, YYYY"
    status: str  # Derived from start_date (Upcoming/Ongoing)
    members: Optional[int]  # No member count in given HTML
    synopsis: Optional[str]  # No synopsis available
    genres: Optional[List[str]]  # Extracted from .genres
    popularity: Optional[int]  # Not available in the provided HTML
    anime_type: Optional[str]  # Extracted from .info
    demographic: Optional[str]  # No demographic info in given HTML
    studios: Optional[str]  # Extracted from .studios


class AnilistSpider(scrapy.Spider):
    name = "anilist_spider"
    table_name = "trending"

    start_urls = [
        "https://anilist.co/search/anime/trending",
    ]
    results: list[ScrapedData] = []

    def start_requests(self):
        for url in self.start_urls:
            yield SeleniumRequest(
                url=url,
                callback=self.parse,
                wait_time=3,
                wait_until=EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "tr.ranking-list")),
                screenshot=True,
            )

    def parse(self, response):
        driver: WebDriver = response.request.meta['driver']

        conn = sqlite3.connect(f"{DATA_DIR}/anime.db")
        cursor = conn.cursor()

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
          next_episode INTEGER,
          next_date TEXT,
          end_date TEXT,
          status TEXT,
          members INTEGER,
          synopsis TEXT,
          genres TEXT,
          popularity INTEGER,
          anime_type TEXT,
          demographic TEXT,
          studios TEXT
      )
      """)

        for index, anime in enumerate(response.css(".results.cover .media-card"), start=1):
            title = anime.css(".title::text").get(default="Unknown").strip()
            relative_url = anime.css(".title").attrib.get("href", "")
            url = normalize_url(urljoin(response.url, relative_url))
            image_url = normalize_url(
                anime.css("img.image::attr(src)").get(""))
            start_date = anime.css(".date::text").get(
                default="Unknown").strip()
            next_airing_text = anime.css(".date.airing::text").get(default="")
            score_text = anime.css(".score .percentage::text").get(
                default="N/A").strip()
            studios = anime.css(".studios::text").get(
                default="Unknown").strip()
            anime_type = anime.css(
                ".info span:first-child::text").get(default="Unknown").strip()
            episodes_text = anime.css(
                ".info span:nth-child(3)::text").get(default="0").strip()
            genres = [genre.get().strip()
                      for genre in anime.css(".genres .genre::text")]

            score = float(score_text.replace("%", "")) / \
                100 if score_text != "N/A" else None
            episodes = int(re.search(r"\d+", episodes_text).group()
                           ) if re.search(r"\d+", episodes_text) else None
            anime_id = self.extract_anime_id(url)

            next_episode = int(re.search(r"Ep (\d+)", next_airing_text).group(1)
                               ) if re.search(r"Ep (\d+)", next_airing_text) else None

            next_date = None
            days_match = re.search(r"in (\d+) days", next_airing_text)
            if days_match:
                days_remaining = int(days_match.group(1))
                next_airing_date = datetime.date.today() + datetime.timedelta(days=days_remaining)
                next_date = next_airing_date.strftime("%B %d, %Y")

            anime_data = ScrapedData(
                id=str(anime_id),
                rank=index,  # ✅ Assign correct rank based on order of .media-card
                title=title,
                url=url,
                image_url=image_url,
                score=score,
                episodes=episodes,
                start_date=start_date,
                next_episode=next_episode or 0,
                next_date=next_date,
                end_date=None,
                status="Ongoing" if next_date else "Finished",
                members=None,
                synopsis=None,
                genres=genres if genres else None,
                popularity=None,
                anime_type=anime_type,
                demographic=None,
                studios=studios,
            )

            try:
                cursor.execute(f"""
              INSERT OR REPLACE INTO {self.table_name} (
                  id, rank, title, url, image_url, score, episodes, start_date, 
                  next_episode, next_date, end_date, status, members, genres, anime_type, studios
              )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              """, (
                    anime_data['id'],
                    anime_data['rank'],  # ✅ Now stored in DB
                    anime_data['title'],
                    anime_data['url'],
                    anime_data['image_url'],
                    anime_data['score'],
                    anime_data['episodes'],
                    anime_data['start_date'],
                    anime_data.get('next_episode', 0),
                    anime_data.get('next_date', None),
                    anime_data['end_date'],
                    anime_data['status'],
                    anime_data['members'],
                    ", ".join(anime_data['genres']) if isinstance(
                        anime_data['genres'], list) else None,
                    anime_data['anime_type'],
                    anime_data['studios']
                ))

            except Exception as e:
                logger.error("Error saving to DB:", exc_info=True)

            self.results.append(anime_data)

        conn.commit()
        conn.close()

        for item in self.results:
            yield item

    def extract_anime_id(self, url: str) -> Optional[str]:
        """Extract the first numeric ID from any Anilist anime URL."""
        match = re.search(r'\/(\d+)', url)  # ✅ Find first numeric segment
        return match.group(1) if match else None  # ✅ Return as string

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
    process.crawl(AnilistSpider)
    process.start()
