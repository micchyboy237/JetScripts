import scrapy
from bs4 import BeautifulSoup
import sqlite3

DB_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper/data"
DB_PATH = f"{DB_DIR}/articles.db"


class ArticleSpider(scrapy.Spider):
    name = "article_spider"
    start_urls = ["https://example-blog.com/"]  # Replace with target website

    def parse(self, response):
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text if soup.find("h1") else "No Title"
        content = "\n".join([p.text for p in soup.find_all("p")])

        if len(content) > 100:  # Filter out noise
            self.store_data(title, content, response.url)

    def store_data(self, title, content, url):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS articles (id INTEGER PRIMARY KEY, title TEXT, content TEXT, url TEXT)")
        cursor.execute(
            "INSERT INTO articles (title, content, url) VALUES (?, ?, ?)", (title, content, url))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess()
    process.crawl(ArticleSpider)
    process.start()
