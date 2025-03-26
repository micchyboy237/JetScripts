import scrapy
import sqlite3
from urllib.parse import quote
from jet.scrapers.browser.scrapy import settings, SeleniumRequest, normalize_url
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from typing import List, TypedDict, Optional

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"


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
    synopsis: Optional[str]
    genres: Optional[List[str]]
    popularity: Optional[int]
    anime_type: Optional[str]
    demographic: Optional[str]


class MyAnimeListSpider(scrapy.Spider):
    name = "myanimelist_spider"
    start_urls = ["https://myanimelist.net/topanime.php"]
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

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS anime (
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
            synopsis TEXT,
            genres TEXT,
            popularity INTEGER,
            anime_type TEXT,
            demographic TEXT
        )
        """)

        for idx, anime in enumerate(response.css("tr.ranking-list")):
            title_element = anime.css(".anime_ranking_h3 a")
            title = title_element.css("::text").get()
            information = anime.css(".information::text").getall()

            episodes_text = information[0].strip() if information else ""
            episodes = int(episodes_text.split(
                "(")[-1].split()[0]) if "(" in episodes_text else None
            duration_text = information[1].strip() if len(
                information) > 1 else ""
            dates = duration_text.split("-")
            start_date = dates[0].strip()
            end_date = dates[1].strip() if len(dates) > 1 else None
            status = "Finished" if end_date else "Ongoing"

            members_text = information[2].strip().replace(
                ",", "") if len(information) > 2 else ""
            members = int(members_text.split()[0]) if members_text.split(
            ) and members_text.split()[0].isdigit() else None

            rank = anime.css("span.top-anime-rank-text::text").get()
            url = quote(
                anime.css("td.title a.hoverinfo_trigger").attrib["href"], safe=":/")
            image_url = quote(
                anime.css("td.title img::attr(data-src)").get(), safe=":/")
            score = anime.css("td.score span.text::text").get()

            rank = int(rank) if rank and rank.isdigit() else None
            score = float(score) if score else None

            url = normalize_url(
                anime.css("td.title a.hoverinfo_trigger").attrib["href"])
            image_url = normalize_url(
                anime.css("td.title img::attr(data-src)").get())

            anime_data = ScrapedData(
                rank=rank,
                title=title.strip() if title else "Unknown",
                url=url,  # ✅ Now stores a clean URL
                image_url=image_url,
                score=score,
                episodes=episodes,
                start_date=start_date,
                end_date=end_date,
                status=status,
                members=members,
            )

            try:
                cursor.execute("""
                INSERT OR IGNORE INTO anime (rank, title, url, image_url, score, episodes, start_date, end_date, status, members)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (anime_data['rank'], anime_data['title'], anime_data['url'],
                      anime_data['image_url'], anime_data['score'], anime_data['episodes'],
                      anime_data['start_date'], anime_data['end_date'], anime_data['status'],
                      anime_data['members']))
            except Exception as e:
                logger.error("Error on saving to DB")
                logger.error(e)

            self.results.append(anime_data)

        conn.commit()
        conn.close()

        for item in self.results:
            yield item

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
    process.crawl(MyAnimeListSpider)
    process.start()
