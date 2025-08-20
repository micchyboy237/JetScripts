from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web.zyte_web.base import ZyteWebReader
from llama_index.readers.zyte_serp import ZyteSerpReader
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
# Zyte Serp Reader

Zyte Serp Reader allows you to access the organic results from google search.  Given a query string, it provides the URLs of the top search results and the text string associated with those.
"""
logger.info("# Zyte Serp Reader")



"""
In this notebook we show how Zyte Serp Reader (along with web reader) can be used collect information about a particular topic. Given these documents we can perform queries on this topic. 

Recently the Govt. of Ireland announced fiscal budget for 2024 and here we show how we can query information regarding the budget. First we get the relevant information using the Zyte Serp Reader, then the information from these URLs is extracted using web reader and finally a queries are answered using a openai chatgpt model.
"""
logger.info("In this notebook we show how Zyte Serp Reader (along with web reader) can be used collect information about a particular topic. Given these documents we can perform queries on this topic.")




zyte_api_key = os.environ.get("ZYTE_API_KEY")

"""
### Get relevant resources (using ZyteSerp)

Given a topic, we use the search results from google to get the links to the relevant pages.
"""
logger.info("### Get relevant resources (using ZyteSerp)")

topic = "Ireland Budget 2025"

serp_reader = ZyteSerpReader(api_key=zyte_api_key)

search_results = serp_reader.load_data(topic)

len(search_results)

for r in search_results[:4]:
    logger.debug(r.text)
    logger.debug(r.metadata)

urls = [r.text for r in search_results]

"""
Seems we have a list of relevant URL with regard to our topic ("Ireland budget 2024"). Metadata also shows the text and rank associated with the search result entry. Next we get the content of these webpages using web reader.

### Get topic content

Given the urls of the webpages which contain information about the topic, we get the content. Since the webpages contain a lot of non-relevant content, we can obtain the filtered content using the "article" mode of the ZyteWebReader which returns only the article content of from the webpage.
"""
logger.info("### Get topic content")

web_reader = ZyteWebReader(api_key=zyte_api_key, mode="article")
documents = web_reader.load_data(urls)

logger.debug(documents[0].text[:200])

len(documents)

"""
### Query engine

# Here a very basic query is performed using VectorStoreIndex. Please make sure that the OPENAI_API_KEY env variable is set before running the following code.
"""
logger.info("### Query engine")


index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query(
    "What kind of energy credits are provided in the budget?"
)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)