from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.oxylabs import OxylabsAmazonProductReader
from llama_index.readers.oxylabs import OxylabsGoogleSearchReader
from llama_index.readers.oxylabs import OxylabsYoutubeTranscriptReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/OxylabsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Oxylabs Reader

Use Oxylabs Reader to get information from Google Search, Amazon and YouTube.
For more information check out the [Oxylabs documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api).
"""
logger.info("# Oxylabs Reader")

# %pip install llama-index llama-index-readers-oxylabs

"""
In this notebook, we show how Oxylabs readers can be used to collect information from different sources.

Firstly, import one of the Oxylabs readers.

Currently available readers are:
* OxylabsAmazonSearchReader
* OxylabsAmazonPricingReader
* OxylabsAmazonProductReader
* OxylabsAmazonSellersReader
* OxylabsAmazonBestsellersReader
* OxylabsAmazonReviewsReader
* OxylabsGoogleSearchReader
* OxylabsGoogleAdsReader
* OxylabsYoutubeTranscriptReader
"""
logger.info("In this notebook, we show how Oxylabs readers can be used to collect information from different sources.")


"""
Instantiate the reader with your username and password.
"""
logger.info("Instantiate the reader with your username and password.")

oxylabs_username = os.environ.get("OXYLABS_USERNAME")
oxylabs_password = os.environ.get("OXYLABS_PASSWORD")

google_search_reader = OxylabsGoogleSearchReader(
    oxylabs_username, oxylabs_password
)

"""
Prepare parameters. This example will load the Google Search results for the 'iPhone 16' query with the 'Berlin, Germany' location.

Check out the [documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api) for more examples.
"""
logger.info("Prepare parameters. This example will load the Google Search results for the 'iPhone 16' query with the 'Berlin, Germany' location.")

results = google_search_reader.load_data(
    {"query": "Iphone 16", "parse": True, "geo_location": "Berlin, Germany"}
)

logger.debug(results[0].text)

"""
## More examples

### Amazon Product
"""
logger.info("## More examples")



amazon_product_reader = OxylabsAmazonProductReader(
    oxylabs_username, oxylabs_password
)

results = amazon_product_reader.load_data(
    {
        "domain": "com",
        "query": "B08D9N7RJ4",
        "parse": True,
        "context": [{"key": "autoselect_variant", "value": True}],
    }
)

logger.debug(results[0].text)

"""
### YouTube Transcript
"""
logger.info("### YouTube Transcript")



youtube_transcript_reader = OxylabsYoutubeTranscriptReader(
    oxylabs_username, oxylabs_password
)

results = youtube_transcript_reader.load_data(
    {
        "query": "SLoqvcnwwN4",
        "context": [
            {"key": "language_code", "value": "en"},
            {"key": "transcript_origin", "value": "uploader_provided"},
        ],
    }
)

logger.debug(results[0].text)

logger.info("\n\n[DONE]", bright=True)