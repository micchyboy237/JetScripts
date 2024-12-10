from typing import List, Dict, Optional, TypedDict, Union
from llama_index.core import SummaryIndex
from llama_index.readers.web import (
    SimpleWebPageReader,
    SpiderWebReader,
    BrowserbaseWebReader,
    FireCrawlWebReader,
    TrafilaturaWebReader,
    RssReader,
    ScrapflyReader,
    ZyteWebReader,
)
from IPython.display import Markdown, display

# Define types


class WebReaderConfig(TypedDict, total=False):
    api_key: str
    mode: str
    params: Optional[Dict]
    scrape_config: Optional[Dict]
    scrape_format: Optional[str]
    download_kwargs: Optional[Dict]
    continue_on_failure: Optional[bool]


class WebReaderResponse(TypedDict):
    documents: List[str]


class WebReaderBase:
    def load_data(self, url: Union[str, List[str]], **kwargs) -> List[str]:
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

# Web Readers


class SimpleReader(WebReaderBase):
    def __init__(self, url: Union[str, List[str]]):
        super().__init__()

        urls = url if isinstance(url, list) else [url]
        self.documents = self.load_data(urls)

    def load_data(self, urls: List[str], html_to_text: bool = True) -> List[str]:
        return SimpleWebPageReader(html_to_text=html_to_text).load_data(urls)

    def query(self, query: str, similarity_top_k: int = 5):
        index = SummaryIndex.from_documents(self.documents, show_progress=True)
        # set Logging to DEBUG for more detailed outputs
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query)
        return response


# class SpiderReader(WebReaderBase):
#     def load_data(self, url: Union[str, List[str]], config: WebReaderConfig) -> List[str]:
#         return SpiderWebReader(api_key=config["api_key"], mode=config["mode"], params=config.get("params")).load_data(url)


# class BrowserbaseReader(WebReaderBase):
#     def load_data(self, url: Union[str, List[str]], text_content: bool = False) -> List[str]:
#         return BrowserbaseWebReader().load_data(urls=[url], text_content=text_content)


# class FireCrawlReader(WebReaderBase):
#     def load_data(self, url: Union[str, List[str]], config: WebReaderConfig) -> List[str]:
#         return FireCrawlWebReader(api_key=config["api_key"], mode=config["mode"], params=config.get("params")).load_data(url)


class TrafilaturaReader(WebReaderBase):
    def __init__(self, url: Union[str, List[str]]):
        super().__init__()

        urls = url if isinstance(url, list) else [url]
        self.documents = self.load_data(urls)

    def load_data(self, urls: List[str]) -> List[str]:
        return TrafilaturaWebReader().load_data(urls, output_format="markdown")

    def query(self, query: str, similarity_top_k: int = 5):
        index = SummaryIndex.from_documents(self.documents, show_progress=True)
        # set Logging to DEBUG for more detailed outputs
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query)
        return response


class RssFeedReader(WebReaderBase):
    def __init__(self, url: Union[str, List[str]]):
        super().__init__()

        urls = url if isinstance(url, list) else [url]
        self.documents = self.load_data(urls)

    def load_data(self, urls: List[str]) -> List[str]:
        return RssReader().load_data(urls)

    def query(self, query: str, similarity_top_k: int = 5):
        index = SummaryIndex.from_documents(self.documents, show_progress=True)
        # set Logging to DEBUG for more detailed outputs
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query)
        return response


# class ScrapflyReaderBase(WebReaderBase):
#     def load_data(self, url: Union[str, List[str]], config: WebReaderConfig) -> List[str]:
#         return ScrapflyReader(
#             api_key=config["api_key"],
#             ignore_scrape_failures=config.get("ignore_scrape_failures", True),
#         ).load_data(urls=[url], scrape_config=config.get("scrape_config"), scrape_format=config.get("scrape_format", "markdown"))


# class ZyteReader(WebReaderBase):
#     def load_data(self, url: Union[str, List[str]], config: WebReaderConfig) -> List[str]:
#         return ZyteWebReader(api_key=config["api_key"], mode=config["mode"], download_kwargs=config.get("download_kwargs"), continue_on_failure=config.get("continue_on_failure", True)).load_data(url)


# Main Function
def main():
    import os
    import json
    from jet.logger import logger
    from jet.file import save_json
    from jet.transformers import make_serializable
    from jet.vectors import SettingsDict, SettingsManager

    settings = SettingsDict(
        llm_model="llama3.1",
        embedding_model="nomic-embed-text",
        chunk_size=768,
        chunk_overlap=50,
        base_url="http://localhost:11434",
    )
    SettingsManager.create(settings)

    similarity_top_k = 3

    generated_dir = "generated/web_page_demo"

    query = "What did the author do growing up?"
    urls = ["http://paulgraham.com/worked.html"]
    rss_urls = ["https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]

    # # Example: SimpleWebPageReader
    # logger.debug("Running SimpleWebPageReader...")
    # reader = SimpleReader(urls)
    # documents = reader.documents
    # save_json(documents, file_path=os.path.join(
    #     generated_dir, "simple_reader_documents.json"))
    # response = reader.query(query, similarity_top_k=similarity_top_k)
    # display(Markdown(f"<b>{response}</b>"))
    # logger.log(f"Simple Reader Output ({len(documents)}):",
    #            json.dumps(make_serializable(documents[0]), indent=2), colors=["LOG", "SUCCESS"])
    # result = {
    #     "query": query,
    #     "response": response,
    # }
    # save_json(result, file_path=os.path.join(
    #     generated_dir, "simple_reader_results.json"))

    # Example: TrafilaturaReader
    logger.debug("Running TrafilaturaReader...")
    reader = TrafilaturaReader(urls)
    documents = reader.documents
    save_json(documents, file_path=os.path.join(
        generated_dir, "trafilatura_reader_documents.json"))
    response = reader.query(query, similarity_top_k=similarity_top_k)
    display(Markdown(f"<b>{response}</b>"))
    logger.log(f"Trafilatura Reader Output ({len(documents)}):",
               json.dumps(make_serializable(documents[0]), indent=2), colors=["LOG", "SUCCESS"])
    result = {
        "query": query,
        "response": response,
    }
    save_json(result, file_path=os.path.join(
        generated_dir, "trafilatura_reader_results.json"))

    # Example: RssReader
    logger.debug("Running RssReader...")
    reader = RssFeedReader(rss_urls)
    documents = reader.documents
    save_json(documents, file_path=os.path.join(
        generated_dir, "rss_reader_documents.json"))
    response = reader.query(query, similarity_top_k=similarity_top_k)
    display(Markdown(f"<b>{response}</b>"))
    logger.log(f"Rss Reader Output ({len(documents)}):",
               json.dumps(make_serializable(documents[0]), indent=2), colors=["LOG", "SUCCESS"])
    result = {
        "query": query,
        "response": response,
    }
    save_json(result, file_path=os.path.join(
        generated_dir, "rss_reader_results.json"))


if __name__ == "__main__":
    main()
