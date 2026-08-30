import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class Crawl4AILoader(BaseLoader):
    def __init__(self, urls: list[str], browser_config: BrowserConfig = None):
        self.urls = urls
        self.browser_config = browser_config or BrowserConfig(headless=True)

    async def aload(self) -> list[Document]:
        docs = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            config = CrawlerRunConfig(word_count_threshold=10)
            results = await crawler.arun_many(urls=self.urls, config=config)
            for result in results:
                if result.success:
                    docs.append(
                        Document(
                            page_content=result.markdown,
                            metadata={"url": result.url, "title": result.title},
                        )
                    )
        return docs


# Usage
async def main():
    loader = Crawl4AILoader(
        [
            "https://docs.langchain.com/oss/deepagents/code/overview",
        ]
    )
    documents = await loader.aload()
    print(f"Loaded {len(documents)} docs")
    print(documents[0].page_content[:300])


asyncio.run(main())
