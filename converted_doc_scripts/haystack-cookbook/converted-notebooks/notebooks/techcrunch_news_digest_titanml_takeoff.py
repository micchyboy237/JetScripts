from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from jet.logger import CustomLogger
from typing import Dict, List
import feedparser
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Getting a quick daily digest from your favorite tech websites

Motivation: We want to stay informed on the latest news in tech. However, with so many websites and news happening every day, it is impossible to keep track of what is going on. But what if we could summarize the latest developments and have all this run locally with an off-the-shelf LLM in a few lines of code?

Let us see how Haystack together with TitanML's Takeoff Inference Server can help us achieve this.

## Run Titan Takeoff Inference Server Image

Remember that you must download this notebook and run it in your local environment. The Titan Takeoff Inference Server allows you to run modern open-source LLMs in your infrastructure.

```bash
docker run --gpus all -e TAKEOFF_MODEL_NAME=TheBloke/Llama-2-7B-Chat-AWQ \
                      -e TAKEOFF_DEVICE=cuda \
                      -e TAKEOFF_MAX_SEQUENCE_LENGTH=256 \
                      -it \
                      -p 3000:3000 tytn/takeoff-pro:0.11.0-gpu
```

## Daily digest from top tech websites using Deepset Haystack and Titan Takeoff
"""
logger.info("# Getting a quick daily digest from your favorite tech websites")

# !pip install feedparser
# !pip install takeoff_haystack




urls = {
    'theverge': 'https://www.theverge.com/rss/frontpage/',
    'techcrunch': 'https://techcrunch.com/feed',
    'mashable': 'https://mashable.com/feeds/rss/all',
    'cnet': 'https://cnet.com/rss/news',
    'engadget': 'https://engadget.com/rss.xml',
    'zdnet': 'https://zdnet.com/news/rss.xml',
    'venturebeat': 'https://feeds.feedburner.com/venturebeat/SZYF',
    'readwrite': 'https://readwrite.com/feed/',
    'wired': 'https://wired.com/feed/rss',
    'gizmodo': 'https://gizmodo.com/rss',
}

NUM_WEBSITES = 3
NUM_TITLES = 1

def get_titles(urls: Dict[str, str], num_sites: int, num_titles: int) -> List[str]:
    titles: List[str] = []
    sites = list(urls.keys())[:num_sites]
    for site in sites:
        feed = feedparser.parse(urls[site])
        entries = feed.entries[:num_titles]
        for entry in entries:
            titles.append(entry.title)
    return titles

titles = get_titles(urls, NUM_WEBSITES, NUM_TITLES)

document_store = InMemoryDocumentStore()
document_store.write_documents(
    [
        Document(content=title) for title in titles
    ]
)

template = """
HEADLINES:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

REQUEST: {{ query }}
"""

pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", TakeoffGenerator(base_url="http://localhost", port=3000))
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

query = f"Summarize each of the {NUM_WEBSITES * NUM_TITLES} provided headlines in three words."

titles_string

response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})

logger.debug(response["llm"]["replies"])

logger.info("\n\n[DONE]", bright=True)