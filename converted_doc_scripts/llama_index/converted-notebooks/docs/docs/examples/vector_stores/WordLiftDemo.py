import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.vector_stores.wordlift import WordliftVectorStore
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import SDO, Namespace, DefinedNamespace
from typing import List, Optional
from wordlift_client import Configuration, ApiClient
from wordlift_client.api.dataset_api import DatasetApi
from wordlift_client.api.entities_api import EntitiesApi
from wordlift_client.api.graph_ql_api import GraphQLApi
from wordlift_client.api.vector_search_queries_api import (
VectorSearchQueriesApi,
)
from wordlift_client.models.graphql_request import GraphqlRequest
from wordlift_client.models.page_vector_search_query_response_item import (
PageVectorSearchQueryResponseItem,
)
from wordlift_client.models.vector_search_query_request import (
VectorSearchQueryRequest,
)
from wordlift_client.rest import ApiException
import advertools as adv
import asyncio
import json
import logging
import os
import pandas as pd
import re
import requests
import shutil
import urllib.parse
import wordlift_client


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
# **WordLift** Vector Store

## Introduction
This script demonstrates how to crawl a product website, extract relevant information, build an SEO-friendly Knowledge Graph (a structured representation of PDPs and PLPs), and leverage it for improved search and user experience.

### Key Features & Libraries:

- Web scraping (Advertools)
- Knowledge Graph creation for Product Detail Pages (PDPs) and Product Listing Pages (PLPs) - WordLift
- Product recommendations (WordLift Neural Search)
- Shopping assistant creation (WordLift + LlamaIndex 🦙)

This approach enhances SEO performance and user engagement for e-commerce sites.

Learn more about how it works here:
- [https://www.youtube.com/watch?v=CH-ir1MTAwQ](https://www.youtube.com/watch?v=CH-ir1MTAwQ)
- [https://wordlift.io/academy-entries/mastering-serp-analysis-knowledge-graphs](https://wordlift.io/academy-entries/mastering-serp-analysis-knowledge-graphs)

</br></br>
<table align="left">
  <td>
  <a href="https://wordlift.io">
    <img width=130px src="https://wordlift.io/wp-content/uploads/2018/07/logo-assets-510x287.png" />
    </a>
    </td>
    <td>
      by
      <a href="https://wordlift.io/blog/en/entity/andrea-volpini">
        Andrea Volpini
      </a>
      and
      <a href="https://wordlift.io/blog/en/entity/david-riccitelli">
        David Riccitelli
      </a>      
      <br/>
      <br/>
      MIT License
      <br/>
      <br/>
      <i>Last updated: <b>Jul 31st, 2024</b></i>
  </td>
</table>
</br>
</br>

# Setup
"""
logger.info("# **WordLift** Vector Store")

# !pip install advertools -q
# !pip install -U wordlift-client # 🎉 first time on stage 🎉
# !pip install rdflib -q


# import nest_asyncio





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# nest_asyncio.apply()

WORDLIFT_KEY = os.getenv("WORDLIFT_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

"""
# Crawl the Website w/ Advertools
"""
logger.info("# Crawl the Website w/ Advertools")

def crawl_website(url, output_file, num_pages=10):
    logger.info(f"Starting crawl of {url}")
    adv.crawl(
        url,
        output_file,
        follow_links=True,
        custom_settings={
            "CLOSESPIDER_PAGECOUNT": num_pages,
            "USER_AGENT": "WordLiftBot/1.0 (Maven Project)",
            "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
            "DOWNLOAD_DELAY": 1,
            "ROBOTSTXT_OBEY": False,
        },
        xpath_selectors={
            "product_description": "/html/body/div[1]/div/div/div/div/div[1]/div/div[3]/div/div[2]/div[2]/div[1]/p/text()",
            "product_price": "/html/body/div[1]/div/div/div/div/div[1]/div/div[3]/div/div[2]/p/span/bdi/text()",
            "product_category": "//span[@class='posted_in']/a/text()",
        },
    )
    logger.info(f"Crawl completed. Results saved to {output_file}")




def analyze_url_patterns(df):
    df["page_type"] = df["url"].apply(
        lambda x: "PLP"
        if "/product-category/" in x
        else ("PDP" if "/product/" in x else "Other")
    )
    logger.info(
        f"Found {(df['page_type'] == 'PLP').sum()} PLPs and {(df['page_type'] == 'PDP').sum()} PDPs"
    )
    return df




def extract_page_data(df):
    extracted_data = []
    for _, row in df.iterrows():
        page = {
            "url": row["url"],
            "title": row["title"],
            "page_type": row["page_type"],
            "meta_description": row.get("meta_description", ""),
            "og_title": row.get("og_title", ""),
            "og_description": row.get("og_description", ""),
            "h1": ", ".join(row.get("h1", []))
            if isinstance(row.get("h1"), list)
            else row.get("h1", ""),
            "h2": ", ".join(row.get("h2", []))
            if isinstance(row.get("h2"), list)
            else row.get("h2", ""),
        }

        if row["page_type"] == "PDP":
            page.update(
                {
                    "product_description": ", ".join(
                        row.get("product_description", [])
                    )
                    if isinstance(row.get("product_description"), list)
                    else row.get("product_description", ""),
                    "product_price": ", ".join(row.get("product_price", []))
                    if isinstance(row.get("product_price"), list)
                    else row.get("product_price", ""),
                    "product_category": ", ".join(
                        row.get("product_category", [])
                    )
                    if isinstance(row.get("product_category"), list)
                    else row.get("product_category", ""),
                }
            )
        elif row["page_type"] == "PLP":
            h1_content = (
                row.get("h1", [""])[0]
                if isinstance(row.get("h1"), list)
                else row.get("h1", "")
            )
            category = (
                h1_content.split("@@")[-1]
                if "@@" in h1_content
                else h1_content.replace("Category: ", "").strip()
            )
            page["category_name"] = category

        extracted_data.append(page)

    return pd.DataFrame(extracted_data)

"""
# Build the KG w/ WordLift 🕸
"""
logger.info("# Build the KG w/ WordLift 🕸")

configuration = Configuration(host="https://api.wordlift.io")
configuration.api_key["ApiKey"] = WORDLIFT_KEY
configuration.api_key_prefix["ApiKey"] = "Key"

EXAMPLE_PRIVATE_NS = Namespace("https://ns.example.org/private/")

BASE_URI = "http://data.wordlift.io/[dataset_id]/"

async def cleanup_knowledge_graph(api_client):
    dataset_api = wordlift_client.DatasetApi(api_client)
    try:
        async def run_async_code_f8d1cbd9():
            await dataset_api.delete_all_entities()
            return 
         = asyncio.run(run_async_code_f8d1cbd9())
        logger.success(format_json())
    except Exception as e:
        logger.debug(
            "Exception when calling DatasetApi->delete_all_entities: %s\n" % e
        )


async def create_entity(entities_api, entity_data):
    g = Graph().parse(data=json.dumps(entity_data), format="json-ld")
    body = g.serialize(format="application/rdf+xml")
    await entities_api.create_or_update_entities(
        body=body, _content_type="application/rdf+xml"
    )


def replace_url(original_url: str) -> str:
    old_domain = "https://product-finder.wordlift.io/"
    new_domain = "https://data-science-with-python-for-seo.wordlift.dev/"

    if original_url.startswith(old_domain):
        return original_url.replace(old_domain, new_domain, 1)
    else:
        return original_url


def create_entity_uri(url):
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path.strip("/")
    path_parts = path.split("/")
    fragment = parsed_url.fragment

    if "product" in path_parts:
        product_id = path_parts[-1]  # Get the last part of the path
        if fragment == "offer":
            return f"{BASE_URI}offer_{product_id}"
        else:
            return f"{BASE_URI}product_{product_id}"
    elif "product-category" in path_parts:
        category = path_parts[-1]  # Get the last part of the path
        return f"{BASE_URI}plp_{category}"
    else:
        safe_path = "".join(c if c.isalnum() else "_" for c in path)
        if fragment == "offer":
            return f"{BASE_URI}offer_{safe_path}"
        else:
            return f"{BASE_URI}page_{safe_path}"


def clean_price(price_str):
    if not price_str or price_str == "N/A":
        return None
    if isinstance(price_str, (int, float)):
        return float(price_str)
    try:
        cleaned_price = "".join(
            char for char in str(price_str) if char.isdigit() or char == "."
        )
        return float(cleaned_price)
    except ValueError:
        logger.warning(f"Could not convert price: {price_str}")
        return None


def create_product_entity(row, dataset_uri):
    url = replace_url(row["url"])
    product_entity_uri = create_entity_uri(url)

    entity_data = {
        "@context": "http://schema.org",
        "@type": "Product",
        "@id": product_entity_uri,
        "url": url,
        "name": row["title"]
        if not pd.isna(row["title"])
        else "Untitled Product",
        "urn:meta:requestEmbeddings": [
            "http://schema.org/name",
            "http://schema.org/description",
        ],
    }

    if not pd.isna(row.get("product_description")):
        entity_data["description"] = row["product_description"]

    if not pd.isna(row.get("product_price")):
        price = clean_price(row["product_price"])
        if price is not None:
            offer_entity_uri = f"{product_entity_uri}/offer_1"
            entity_data["offers"] = {
                "@type": "Offer",
                "@id": offer_entity_uri,
                "price": str(price),
                "priceCurrency": "GBP",
                "availability": "http://schema.org/InStock",
                "url": url,
            }

    if not pd.isna(row.get("product_category")):
        entity_data["category"] = row["product_category"]

    custom_attributes = {
        key: row[key]
        for key in [
            "meta_description",
            "og_title",
            "og_description",
            "h1",
            "h2",
        ]
        if not pd.isna(row.get(key))
    }
    if custom_attributes:
        entity_data[str(EXAMPLE_PRIVATE_NS.attributes)] = json.dumps(
            custom_attributes
        )

    return entity_data


def create_collection_entity(row, dataset_uri):
    url = replace_url(row["url"])
    entity_uri = create_entity_uri(url)

    entity_data = {
        "@context": "http://schema.org",
        "@type": "CollectionPage",
        "@id": entity_uri,
        "url": url,
        "name": row["category_name"] or row["title"],
    }

    custom_attributes = {
        key: row[key]
        for key in [
            "meta_description",
            "og_title",
            "og_description",
            "h1",
            "h2",
        ]
        if row.get(key)
    }
    if custom_attributes:
        entity_data[str(EXAMPLE_PRIVATE_NS.attributes)] = json.dumps(
            custom_attributes
        )

    return entity_data


async def build_knowledge_graph(df, dataset_uri, api_client):
    entities_api = EntitiesApi(api_client)

    for _, row in df.iterrows():
        try:
            if row["page_type"] == "PDP":
                entity_data = create_product_entity(row, dataset_uri)
            elif row["page_type"] == "PLP":
                entity_data = create_collection_entity(row, dataset_uri)
            else:
                logger.warning(
                    f"Skipping unknown page type for URL: {row['url']}"
                )
                continue

            if entity_data is None:
                logger.warning(
                    f"Skipping page due to missing critical data: {row['url']}"
                )
                continue

            async def run_async_code_51abf54c():
                await create_entity(entities_api, entity_data)
                return 
             = asyncio.run(run_async_code_51abf54c())
            logger.success(format_json())
            logger.info(
                f"Created entity for {row['page_type']}: {row['title']}"
            )
        except Exception as e:
            logger.error(
                f"Error creating entity for {row['page_type']}: {row['title']}"
            )
            logger.error(f"Error: {str(e)}")

"""
# Run the show
"""
logger.info("# Run the show")

CRAWL_URL = "https://product-finder.wordlift.io/"
OUTPUT_FILE = "crawl_results.jl"


async def main():
    crawl_website(CRAWL_URL, OUTPUT_FILE)

    df = pd.read_json(OUTPUT_FILE, lines=True)

    df = analyze_url_patterns(df)

    pages_df = extract_page_data(df)

    async def async_func_13():
        async with ApiClient(configuration) as api_client:
            try:
                await cleanup_knowledge_graph(api_client)
                logger.info(f"Knowledge Graph Cleaned Up")
            except Exception as e:
                logger.error(
                    f"Failed to clean up the existing Knowledge Graph: {str(e)}"
                )
                return  # Exit if cleanup fails
            
            await build_knowledge_graph(pages_df, CRAWL_URL, api_client)
            
        return result

    result = asyncio.run(async_func_13())
    logger.success(format_json(result))
    logger.info("Knowledge graph building completed.")


if __name__ == "__main__":
    asyncio.run(main())

"""
## Let's query products in the KG now using GraphQL
"""
logger.info("## Let's query products in the KG now using GraphQL")

async def perform_graphql_query(api_client):
    graphql_api = GraphQLApi(api_client)
    query = """
    {
        products(rows: 20) {
            id: iri
            category: string(name:"schema:category")
            name: string(name:"schema:name")
            description: string(name:"schema:description")
            url: string(name:"schema:url")
        }
    }
    """
    request = GraphqlRequest(query=query)

    try:
        async def run_async_code_27db5e3d():
            async def run_async_code_c3ab376d():
                response = await graphql_api.graphql_using_post(body=request)
                return response
            response = asyncio.run(run_async_code_c3ab376d())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_27db5e3d())
        logger.success(format_json(response))
        logger.debug("GraphQL Query Results:")
        logger.debug(json.dumps(response, indent=2))
    except Exception as e:
        logger.error(f"An error occurred during GraphQL query: {e}")


async def async_func_23():
    async with ApiClient(configuration) as api_client:
        await perform_graphql_query(api_client)
        logger.info("Knowledge graph building and GraphQL query completed.")
    return result

result = asyncio.run(async_func_23())
logger.success(format_json(result))

"""
# Leveraging the Knowledge Graph

Now that we have successfully created a Knowledge Graph for our e-commerce website, complete with product embeddings, we can take advantage of it to enhance user experience and functionality. The embeddings we've generated for each product allow us to perform semantic similarity searches and build more intelligent systems.

## Adding Structured Data to your Web Pages

In this section, we will perform a simple test of WordLift's data API. This API is used to inject structured data markup from the Knowledge Graph (KG) into your webpages. Structured data helps search engines better understand your content, potentially leading to rich snippets in search results and improved SEO.

For this notebook, we're using a pre-configured KG on a demo e-commerce website. We'll be referencing a fictitious URL: `https://data-science-with-python-for-seo.wordlift.dev`.

When calling WordLift's data API, we simply pass a URL and receive the corresponding JSON-LD (JavaScript Object Notation for Linked Data). This structured data typically includes information such as product details, pricing, and availability for e-commerce sites.

The `get_json_ld_from_url()` function below demonstrates this process. It takes a URL as input and returns the structured data in JSON-LD format, ready to be injected into your webpage.
"""
logger.info("# Leveraging the Knowledge Graph")

def get_json_ld_from_url(url):
    api_url = "https://api.wordlift.io/data/https/" + url.replace(
        "https://", ""
    )

    response = requests.get(api_url)

    if response.status_code == 200:
        json_ld = response.json()
        return json_ld
    else:
        logger.debug(f"Failed to retrieve data: {response.status_code}")
        return None


def pretty_print_json(json_obj):
    logger.debug(json.dumps(json_obj, indent=4))

url = "https://data-science-with-python-for-seo.wordlift.dev/product/100-pure-deluxe-travel-pack-duo-2/"
json_ld = get_json_ld_from_url(url)
json_ld

"""
## Generating Links of Similar Products using WordLift Neural Search

With our product embeddings in place, we can now leverage WordLift's Neural Search capabilities to recommend similar products to users. This feature significantly enhances user engagement and can potentially boost sales by showcasing relevant products based on semantic similarity.

Unlike traditional keyword matching, semantic similarity considers the context and meaning of product descriptions. This approach allows for more nuanced and accurate recommendations, even when products don't share exact keywords.

The `get_top_k_similar_urls` function we've defined earlier implements this functionality. It takes a product URL and returns a list of semantically similar products, ranked by their similarity scores.

For example, if a user is viewing a red cotton t-shirt, this feature might recommend other cotton t-shirts in different colors, or similar style tops made from different materials. This creates a more intuitive and engaging shopping experience for the user.

By implementing this Neural Search feature, we're able to create a more personalized and efficient shopping experience, potentially leading to increased user satisfaction and higher conversion rates.
"""
logger.info("## Generating Links of Similar Products using WordLift Neural Search")

async def get_top_k_similar_urls(configuration, query_url: str, top_k: int):
    request = VectorSearchQueryRequest(
        query_url=query_url,
        similarity_top_k=top_k,
    )

    async def async_func_6():
        async with wordlift_client.ApiClient(configuration) as api_client:
            api_instance = VectorSearchQueriesApi(api_client)
            try:
                page = await api_instance.create_query(
                    vector_search_query_request=request
                )
                return [
                    {
                        "url": item.id,
                        "name": item.text.split("\n")[0],
                        "score": item.score,
                    }
                    for item in page.items
                    if item.id and item.text
                ]
            except Exception as e:
                logger.error(f"Error querying for entities: {e}", exc_info=True)
                return None
            
        return result

    result = asyncio.run(async_func_6())
    logger.success(format_json(result))
top_k = 10
url = "https://data-science-with-python-for-seo.wordlift.dev/product/100-mineral-sunscreen-spf-30/"
async def async_func_27():
    similar_urls = await get_top_k_similar_urls(
        configuration, query_url=url, top_k=top_k
    )
    return similar_urls
similar_urls = asyncio.run(async_func_27())
logger.success(format_json(similar_urls))
logger.debug(json.dumps(similar_urls, indent=2))

"""
## Building a Chatbot for the E-commerce Website using LlamaIndex 🦙

The Knowledge Graph we've created serves as a perfect foundation for building an intelligent chatbot. LlamaIndex (formerly GPT Index) is a powerful data framework that allows us to ingest, structure, and access private or domain-specific data in Large Language Models (LLMs). With LlamaIndex, we can create a context-aware chatbot that understands our product catalog and can assist customers effectively.

By leveraging LlamaIndex in conjunction with our Knowledge Graph, we can develop a chatbot that responds to direct queries. This chatbot will have an understanding of the product catalog, enabling it to:

1. Answer questions about product specifications, availability, and pricing
2. Make personalized product recommendations based on customer preferences
3. Provide comparisons between similar products

This approach leads to more natural and helpful interactions with customers, enhancing their shopping experience. The chatbot can draw upon the structured data in our Knowledge Graph, using LlamaIndex to efficiently retrieve and present relevant information through the LLM.

In the following sections, we'll walk through the process of setting up LlamaIndex with our Knowledge Graph data and creating a chatbot that can intelligently assist our e-commerce customers.

### Installing `LlamaIndex` and `WordLiftVectorStore` 💪
"""
logger.info("## Building a Chatbot for the E-commerce Website using LlamaIndex 🦙")

# %%capture
# !pip install llama-index
# !pip install -U 'git+https://github.com/wordlift/llama_index.git#egg=llama-index-vector-stores-wordlift&subdirectory=llama-index-integrations/vector_stores/llama-index-vector-stores-wordlift'
# !pip install llama-index-embeddings-nomic


"""
### Setting NomicEmbeddings for our Query Engine

Nomic has released v1.5 🪆🪆🪆 of their embedding model, which brings significant improvements to text embedding capabilities. Embeddings are numerical representations of text that capture semantic meaning, allowing our system to understand and compare the content of queries and documents.

Key features of **Nomic v1.5** include:

- Variable-sized embeddings with dimensions between 64 and 768
- Matryoshka learning, which allows for nested representations
- An expanded context size of 8192 tokens

We use NomicEmbeddings in WordLift due to these advanced features, and now we're configuring LlamaIndex to use it as well when encoding user queries. This consistency in embedding models across our stack ensures better alignment between our Knowledge Graph and the query understanding process.

More information on NomicEmbeddings can be found [here](https://www.nomic.ai/blog/posts/nomic-embed-matryoshka).

Go here to [get your free key](https://atlas.nomic.ai/).
"""
logger.info("### Setting NomicEmbeddings for our Query Engine")


nomic_api_key = os.getenv("NOMIC_KEY")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=128,
    model_name="nomic-embed-text-v1.5",
)

embedding = embed_model.get_text_embedding("Hey Ho SEO!")
len(embedding)

"""
We will use MLX as default LLM for generating response. We could of course use any other available LLM.
"""
logger.info("We will use MLX as default LLM for generating response. We could of course use any other available LLM.")

# os.environ["OPENAI_API_KEY"] = OPENAI_KEY

"""
Let's setup now WordliftVectorStore using data from our Knowledge Graph.
"""
logger.info("Let's setup now WordliftVectorStore using data from our Knowledge Graph.")

vector_store = WordliftVectorStore(key=API_KEY)

index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=embed_model
)

query_engine = index.as_query_engine()

query1 = "Can you give me a product similar to the facial puff? Please add the URL also"
result1 = query_engine.query(query1)

logger.debug(result1)

def query_engine(query):
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response


while True:
    user_query = input("Enter your query (or 'quit' to exit): ")
    if user_query.lower() == "quit":
        break
    result = query_engine(user_query)
    logger.debug(result)
    logger.debug("\n---\n")

logger.info("\n\n[DONE]", bright=True)