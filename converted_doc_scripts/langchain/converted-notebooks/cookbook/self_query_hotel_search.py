from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)
from langchain.retrievers import SelfQueryRetriever
from langchain_elasticsearch import ElasticsearchStore
import json
import os
import pandas as pd
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
# Building hotel room search with self-querying retrieval

In this example we'll walk through how to build and iterate on a hotel room search service that leverages an LLM to generate structured filter queries that can then be passed to a vector store.

For an introduction to self-querying retrieval [check out the docs](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query).

## Imports and data prep

In this example we use `ChatOllama` for the model and `ElasticsearchStore` for the vector store, but these can be swapped out with an LLM/ChatModel and [any VectorStore that support self-querying](https://python.langchain.com/docs/integrations/retrievers/self_query/).

Download data from: https://www.kaggle.com/datasets/keshavramaiah/hotel-recommendation
"""
logger.info("# Building hotel room search with self-querying retrieval")

# !pip install langchain langchain-elasticsearch lark ollama elasticsearch pandas


details = (
    pd.read_csv("~/Downloads/archive/Hotel_details.csv")
    .drop_duplicates(subset="hotelid")
    .set_index("hotelid")
)
attributes = pd.read_csv(
    "~/Downloads/archive/Hotel_Room_attributes.csv", index_col="id"
)
price = pd.read_csv("~/Downloads/archive/hotels_RoomPrice.csv", index_col="id")

latest_price = price.drop_duplicates(subset="refid", keep="last")[
    [
        "hotelcode",
        "roomtype",
        "onsiterate",
        "roomamenities",
        "maxoccupancy",
        "mealinclusiontype",
    ]
]
latest_price["ratedescription"] = attributes.loc[latest_price.index]["ratedescription"]
latest_price = latest_price.join(
    details[["hotelname", "city", "country", "starrating"]], on="hotelcode"
)
latest_price = latest_price.rename(
    {"ratedescription": "roomdescription"}, axis=1)
latest_price["mealsincluded"] = ~latest_price["mealinclusiontype"].isnull()
latest_price.pop("hotelcode")
latest_price.pop("mealinclusiontype")
latest_price = latest_price.reset_index(drop=True)
latest_price.head()

"""
## Describe data attributes

We'll use a self-query retriever, which requires us to describe the metadata we can filter on.

Or if we're feeling lazy we can have a model write a draft of the descriptions for us :)
"""
logger.info("## Describe data attributes")


model = ChatOllama(model="llama3.2")
res = model.predict(
    "Below is a table with information about hotel rooms. "
    "Return a JSON list with an entry for each column. Each entry should have "
    '{"name": "column name", "description": "column description", "type": "column data type"}'
    f"\n\n{latest_price.head()}\n\nJSON:\n"
)


attribute_info = json.loads(res)
attribute_info

"""
For low cardinality features, let's include the valid values in the description
"""
logger.info(
    "For low cardinality features, let's include the valid values in the description")

latest_price.nunique()[latest_price.nunique() < 40]

attribute_info[-2]["description"] += (
    f". Valid values are {sorted(latest_price['starrating'].value_counts().index.tolist())}"
)
attribute_info[3]["description"] += (
    f". Valid values are {sorted(latest_price['maxoccupancy'].value_counts().index.tolist())}"
)
attribute_info[-3]["description"] += (
    f". Valid values are {sorted(latest_price['country'].value_counts().index.tolist())}"
)

attribute_info

"""
## Creating a query constructor chain

Let's take a look at the chain that will convert natural language requests into structured queries.

To start we can just load the prompt and see what it looks like
"""
logger.info("## Creating a query constructor chain")


doc_contents = "Detailed description of a hotel room"
prompt = get_query_constructor_prompt(doc_contents, attribute_info)
logger.debug(prompt.format(query="{query}"))

chain = load_query_constructor_runnable(
    ChatOllama(model="llama3.2"), doc_contents, attribute_info
)

chain.invoke(
    {"query": "I want a hotel in Southern Europe and my budget is 200 bucks."})

chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)

"""
## Refining attribute descriptions

We can see that at least two issues above. First is that when we ask for a Southern European destination we're only getting a filter for Italy, and second when we ask for AC we get a literal string lookup for AC (which isn't so bad but will miss things like 'Air conditioning').

As a first step, let's try to update our description of the 'country' attribute to emphasize that equality should only be used when a specific country is mentioned.
"""
logger.info("## Refining attribute descriptions")

attribute_info[-3]["description"] += (
    ". NOTE: Only use the 'eq' operator if a specific country is mentioned. If a region is mentioned, include all relevant countries in filter."
)
chain = load_query_constructor_runnable(
    ChatOllama(model="llama3.2"),
    doc_contents,
    attribute_info,
)

chain.invoke(
    {"query": "I want a hotel in Southern Europe and my budget is 200 bucks."})

"""
## Refining which attributes to filter on

This seems to have helped! Now let's try to narrow the attributes we're filtering on. More freeform attributes we can leave to the main query, which is better for capturing semantic meaning than searching for specific substrings.
"""
logger.info("## Refining which attributes to filter on")

content_attr = ["roomtype", "roomamenities", "roomdescription", "hotelname"]
doc_contents = "A detailed description of a hotel room, including information about the room type and room amenities."
filter_attribute_info = tuple(
    ai for ai in attribute_info if ai["name"] not in content_attr
)
chain = load_query_constructor_runnable(
    ChatOllama(model="llama3.2"),
    doc_contents,
    filter_attribute_info,
)

chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)

"""
## Adding examples specific to our use case

We've removed the strict filter for 'AC' but it's still not being included in the query string. Our chain prompt is a few-shot prompt with some default examples. Let's see if adding use case-specific examples will help:
"""
logger.info("## Adding examples specific to our use case")

examples = [
    (
        "I want a hotel in the Balkans with a king sized bed and a hot tub. Budget is $300 a night",
        {
            "query": "king-sized bed, hot tub",
            "filter": 'and(in("country", ["Bulgaria", "Greece", "Croatia", "Serbia"]), lte("onsiterate", 300))',
        },
    ),
    (
        "A room with breakfast included for 3 people, at a Hilton",
        {
            "query": "Hilton",
            "filter": 'and(eq("mealsincluded", true), gte("maxoccupancy", 3))',
        },
    ),
]
prompt = get_query_constructor_prompt(
    doc_contents, filter_attribute_info, examples=examples
)
logger.debug(prompt.format(query="{query}"))

chain = load_query_constructor_runnable(
    ChatOllama(model="llama3.2"),
    doc_contents,
    filter_attribute_info,
    examples=examples,
)

chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)

"""
This seems to have helped! Let's try another complex query:
"""
logger.info("This seems to have helped! Let's try another complex query:")

chain.invoke(
    {
        "query": "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
    }
)

"""
## Automatically ignoring invalid queries

It seems our model get's tripped up on this more complex query and tries to search over an attribute ('description') that doesn't exist. By setting `fix_invalid=True` in our query constructor chain, we can automatically remove any parts of the filter that is invalid (meaning it's using disallowed operations, comparisons or attributes).
"""
logger.info("## Automatically ignoring invalid queries")

chain = load_query_constructor_runnable(
    ChatOllama(model="llama3.2"),
    doc_contents,
    filter_attribute_info,
    examples=examples,
    fix_invalid=True,
)

chain.invoke(
    {
        "query": "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
    }
)

"""
## Using with a self-querying retriever

Now that our query construction chain is in a decent place, let's try using it with an actual retriever. For this example we'll use the [ElasticsearchStore](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch).
"""
logger.info("## Using with a self-querying retriever")


embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
## Populating vectorstore

The first time you run this, uncomment the below cell to first index the data.
"""
logger.info("## Populating vectorstore")


vecstore = ElasticsearchStore(
    "hotel_rooms",
    embedding=embeddings,
    es_url="http://localhost:9200",
)


retriever = SelfQueryRetriever(
    query_constructor=chain, vectorstore=vecstore, verbose=True
)

results = retriever.invoke(
    "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
)
for res in results:
    logger.debug(res.page_content)
    logger.debug("\n" + "-" * 20 + "\n")

logger.info("\n\n[DONE]", bright=True)
