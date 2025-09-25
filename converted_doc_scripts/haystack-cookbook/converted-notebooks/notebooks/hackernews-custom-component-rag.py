from haystack import Pipeline
from haystack import component, Document
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from jet.logger import logger
from newspaper import Article
from typing import List
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Building a custom component for RAG pipelines with Haystack

*by Tuana Celik: [Twitter](https://twitter.com/tuanacelik), [LinkedIn](https://www.linkedin.com/in/tuanacelik/)*

📚 Check out the [**Customizing RAG Pipelines to Summarize Latest Hacker News Posts with Haystack**](https://haystack.deepset.ai/blog/customizing-rag-to-summarize-hacker-news-posts-with-haystack2) article for a detailed run through of this example.

### Install dependencies
"""
logger.info("# Building a custom component for RAG pipelines with Haystack")

# !pip install newspaper3k
# !pip install haystack-ai

"""
## Create a Custom Haystack Component

This `HackernewsNewestFetcher` ferches the `last_k` newest posts on Hacker News and returns the contents as a List of Haystack Document objects
"""
logger.info("## Create a Custom Haystack Component")


@component
class HackernewsNewestFetcher():

  @component.output_types(articles=List[Document])
  def run(self, last_k: int):
    newest_list = requests.get(url='https://hacker-news.firebaseio.com/v0/newstories.json?print=pretty')
    articles = []
    for id in newest_list.json()[0:last_k]:
      article = requests.get(url=f"https://hacker-news.firebaseio.com/v0/item/{id}.json?print=pretty")
      if 'url' in article.json():
        articles.append(article.json()['url'])

    docs = []
    for url in articles:
      try:
        article = Article(url)
        article.download()
        article.parse()
        docs.append(Document(content=article.text, meta={'title': article.title, 'url': url}))
      except:
        logger.debug(f"Couldn't download {url}, skipped")
    return {'articles': docs}

"""
## Create a Haystack 2.0 RAG Pipeline

This pipeline uses the components available in the Haystack 2.0 preview package at time of writing (22 September 2023) as well as the custom component we've created above.

The end result is a RAG pipeline designed to provide a list of summaries for each of the `last_k` posts on Hacker News, followes by the source URL.
"""
logger.info("## Create a Haystack 2.0 RAG Pipeline")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Ollama Key: ")


prompt_template = """
You will be provided a few of the latest posts in HackerNews, followed by their URL.
For each post, provide a brief summary followed by the URL the full post can be found in.

Posts:
{% for article in articles %}
  {{article.content}}
  URL: {{article.meta['url']}}
{% endfor %}
"""

prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(model="llama3.2")
fetcher = HackernewsNewestFetcher()

pipe = Pipeline()
pipe.add_component("hackernews_fetcher", fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("hackernews_fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

result = pipe.run(data={"hackernews_fetcher": {"last_k": 3}})
logger.debug(result['llm']['replies'][0])

logger.info("\n\n[DONE]", bright=True)