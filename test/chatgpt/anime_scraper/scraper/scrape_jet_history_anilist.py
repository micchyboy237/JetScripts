import datetime
import json
import random
import re

from tqdm import tqdm
import scrapy
import psycopg2
from urllib.parse import quote, urlencode, urljoin
from jet.scrapers.browser.scrapy import settings, SeleniumRequest, normalize_url
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from jet.logger import logger, sleep_countdown
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from typing import List, TypedDict, Optional

ANIME_TITLES_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data/aniwatch_history.json"

DB_CONFIG = {
    "dbname": "anime_db1",
    "user": "jethroestrada",
    "password": "",
    "host": "Jethros-MacBook-Air.local",
    "port": "5432"
}


class ScrapedData(TypedDict):
    id: str
    rank: Optional[int]
    title: str
    url: str
    image_url: str
    score: Optional[float]
    episodes: Optional[int]
    start_date: str
    end_date: Optional[str]
    next_episode: Optional[int]
    next_date: Optional[str]
    members: Optional[int]
    synopsis: Optional[str]
    genres: Optional[List[str]]
    popularity: Optional[int]
    anime_type: Optional[str]
    demographic: Optional[str]
    studios: Optional[str]


class AnilistHistorySpider(scrapy.Spider):
    name = "anilist_history_spider"
    table_name = "history"

    results: list[ScrapedData] = []

    def start_requests(self):
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch anime URLs that need updates
        cursor.execute(f"""
            SELECT title FROM {self.table_name}
        """)
        anime_data = [(row[0].strip())
                      for row in cursor.fetchall()]
        conn.close()

        # Load anime titles from the JSON file
        with open(ANIME_TITLES_PATH, "r", encoding="utf-8") as file:
            anime_titles = json.load(file)

        anime_titles = [
            title for title in anime_titles if not any(t.strip().lower().startswith(title.lower().strip()) for t in anime_data)]

        for title in tqdm(anime_titles, desc="Scraping search results..."):
            # Delay to prevent spam
            delay = random.uniform(1, 4)
            sleep_countdown(delay, f"Delaying:")

            # Construct search URL
            # query_params = {"q": title}
            # search_url = f"https://myanimelist.net/search/all?{urlencode(query_params)}"
            query_params = {"search": title}
            search_url = f"https://anilist.co/search/anime?{urlencode(query_params)}"

            yield SeleniumRequest(
                url=search_url,
                callback=self.parse,
                wait_time=3,
                wait_until=EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".results.chart")),
                screenshot=True,
            )

    def parse(self, response):
        driver: WebDriver = response.request.meta['driver']

        conn = psycopg2.connect(**DB_CONFIG)
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
                rank=index,
                title=title,
                url=url,
                image_url=image_url,
                score=score,
                episodes=episodes,
                start_date=start_date,
                next_episode=next_episode or 0,
                next_date=next_date,
                end_date=None,
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
                INSERT INTO {self.table_name} (
                    id, rank, title, url, image_url, score, episodes, start_date, 
                    next_episode, next_date, end_date, members, genres, anime_type, studios
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    rank = COALESCE(EXCLUDED.rank, {self.table_name}.rank),
                    title = COALESCE(EXCLUDED.title, {self.table_name}.title),
                    url = COALESCE(EXCLUDED.url, {self.table_name}.url),
                    image_url = COALESCE(EXCLUDED.image_url, {self.table_name}.image_url),
                    score = COALESCE(EXCLUDED.score, {self.table_name}.score),
                    episodes = COALESCE(EXCLUDED.episodes, {self.table_name}.episodes),
                    start_date = COALESCE(EXCLUDED.start_date, {self.table_name}.start_date),
                    next_episode = COALESCE(EXCLUDED.next_episode, {self.table_name}.next_episode),
                    next_date = COALESCE(EXCLUDED.next_date, {self.table_name}.next_date),
                    end_date = COALESCE(EXCLUDED.end_date, {self.table_name}.end_date),
                    members = COALESCE(EXCLUDED.members, {self.table_name}.members),
                    genres = COALESCE(EXCLUDED.genres, {self.table_name}.genres),
                    anime_type = COALESCE(EXCLUDED.anime_type, {self.table_name}.anime_type),
                    studios = COALESCE(EXCLUDED.studios, {self.table_name}.studios);
                """, (
                    anime_data['id'],
                    anime_data['rank'],
                    anime_data['title'],
                    anime_data['url'],
                    anime_data['image_url'],
                    anime_data['score'],
                    anime_data['episodes'],
                    anime_data['start_date'],
                    anime_data.get('next_episode', 0),
                    anime_data.get('next_date', None),
                    anime_data['end_date'],
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
        cursor.close()
        conn.close()

        for item in self.results:
            yield item

    def extract_anime_id(self, url: str) -> Optional[str]:
        match = re.search(r'\/(\d+)', url)
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
    process.crawl(AnilistHistorySpider)
    process.start()
