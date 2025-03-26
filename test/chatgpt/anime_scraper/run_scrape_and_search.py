
# Ensure the project directory is added to sys.path
import sys

from jet.executor.command import run_command

project_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper"
sys.path.insert(0, project_path)

# Run scrapers
if __name__ == "__main__":
    import jet.scrapers.browser.scrapy.settings.config
    from scrapy.crawler import CrawlerProcess
    from scraper.scrape_mal import MyAnimeListSpider
    from scraper.scrape_mal_details import MyAnimeDetailsSpider
    from scraper.scrape_anilist import AniListSpider

    process = CrawlerProcess()
    process.crawl(MyAnimeListSpider)
    process.crawl(MyAnimeDetailsSpider)
    # process.crawl(AniListSpider)
    process.start()

# Run RAG search
if __name__ == "__main__":
    run_command("python rag.py")
