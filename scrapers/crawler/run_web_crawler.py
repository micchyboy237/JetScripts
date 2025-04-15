from jet.scrapers.crawler.web_crawler import WebCrawler, sort_urls_numerically
from jet.file.utils import save_data
from jet.logger import logger

# Example usage
if __name__ == "__main__":
    urls = [
        "https://www.justwatch.com/us/tv-show/ill-become-a-villainess-that-will-go-down-in-history-the-more-of-a-villainess-i-become-the-more-the-prince-will-dote-on-me",
        "https://en.wikipedia.org/wiki/I%27ll_Become_a_Villainess_Who_Goes_Down_in_History",
        "https://www.imdb.com/title/tt32812118/",
        "https://reelgood.com/show/ill-become-a-villainess-who-goes-down-in-history-2024",
        "https://www.crunchyroll.com/series/GQWH0M17X/ill-become-a-villainess-who-goes-down-in-history",
        "https://www.primevideo.com/detail/I%E2%80%99ll-Become-a-Villainess-Who-Goes-Down-in-History/0TXGYPVOUNWCF3MHBTVH493HSK",
        "https://myanimelist.net/anime/56228/Rekishi_ni_Nokoru_Akujo_ni_Naru_zo",
        "https://www.anime-planet.com/anime/ill-become-a-villainess-who-goes-down-in-history",
        "https://www.bilibili.tv/video/4794017678102528"
    ]

    includes_all = ["*villainess*", "*down*", "*history*"]
    excludes = []
    max_depth = None

    crawler = WebCrawler(urls=urls, excludes=excludes,
                         includes_all=includes_all, max_depth=max_depth)

    output_file = f"generated/crawl/crawled_urls.json"
    batch_size = 5
    batch_count = 0

    results = []
    for result in crawler.crawl():
        logger.info(
            f"Saving {len(crawler.passed_urls)} pages to {output_file}")
        results.append(result)
        save_data(output_file, results, write=True)

    crawler.close()
