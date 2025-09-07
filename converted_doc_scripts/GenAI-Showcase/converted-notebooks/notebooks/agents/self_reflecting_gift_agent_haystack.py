from bson import json_util
from colorama import Fore
from haystack import Document, Pipeline
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import OllamaDocumentEmbedder
from haystack.components.embedders import OllamaTextEmbedder
from haystack.components.generators import OllamaGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.mongodb_atlas import (
MongoDBAtlasEmbeddingRetriever,
)
from haystack_integrations.document_stores.mongodb_atlas import (
MongoDBAtlasDocumentStore,
)
from jet.logger import CustomLogger
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from typing import List
import os
import re
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Self-Reflecting Gift Agent with Haystack and MongoDB Atlas
This notebook demonstrates how to build a self-reflecting gift selection agent using [Haystack](https://haystack.deepset.ai/) and MongoDB Atlas!

The agent will help optimize gift selections based on children's wishlists and budget constraints, using MongoDB Atlas vector search for semantic matching and implementing self-reflection to ensure the best possible gift combinations.

**Components to use in this notebook:**
- [`OllamaTextEmbedder`](https://docs.haystack.deepset.ai/docs/ollamatextembedder) for  query embedding
- [`MongoDBAtlasEmbeddingRetriever`](https://docs.haystack.deepset.ai/docs/) for finding relevant gifts
- [`PromptBuilder`](https://docs.haystack.deepset.ai/docs/promptbuilder) for creating the prompt
- [`OllamaGenerator`](https://docs.haystack.deepset.ai/docs/ollamagenerator) for  generating responses
- Custom `GiftChecker` component for self-reflection

### **Prerequisites**

Before running this notebook, you'll need:

* A MongoDB Atlas account and cluster
* Python environment with `haystack-ai`, `mongodb-atlas-haystack` and other required packages
* Ollama API key for GPT-4 and `mxbai-embed-large` access
"""
logger.info("# Self-Reflecting Gift Agent with Haystack and MongoDB Atlas")

# !pip install haystack-ai mongodb-atlas-haystack tiktoken datasets colorama

"""
## Configure Environment

* Create a free MongoDB Atlas account at https://www.mongodb.com/cloud/atlas/register
* Create a new cluster (free tier is sufficient). Find more details in [this tutorial](https://www.mongodb.com/docs/guides/atlas/cluster/#create-a-cluster)
* Create a database user with read/write permissions
* Get your [connection string](https://www.mongodb.com/docs/atlas/tutorial/connect-to-your-cluster/#connect-to-your-atlas-cluster) from Atlas UI (Click "Connect" > "Connect your application")
* Connection string should look like this `mongodb+srv://<db_username>:<db_password>@<clustername>.xxxxx.mongodb.net/?retryWrites=true...`. Replace `<db_password>` in the connection string with your database user's password
* Enable network access from your IP address in the Network Access settings (have `0.0.0.0/0` address in your network access list).

Set up your MongoDB Atlas and Ollama credentials:
"""
logger.info("## Configure Environment")

# import getpass

# conn_str = getpass.getpass("Enter your MongoDB connection string:")
conn_str = (
    re.sub(r"appName=[^\s]*", "appName=devrel.ai.haystack_partner", conn_str)
    if "appName=" in conn_str
    else conn_str
    + ("&" if "?" in conn_str else "?")
    + "appName=devrel.ai.haystack_partner"
)
os.environ["MONGO_CONNECTION_STRING"] = conn_str
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API Key:")

"""
## Create Sample Gift Dataset

Let's create a dataset of gifts with prices and categories:
"""
logger.info("## Create Sample Gift Dataset")

dataset = {
    "train": [
        {
            "title": "LEGO Star Wars Set",
            "price": "$49.99",
            "description": "Build your own galaxy with this exciting LEGO Star Wars set",
            "category": "Toys",
            "age_range": "7-12",
        },
        {
            "title": "Remote Control Car",
            "price": "$29.99",
            "description": "Fast and fun RC car with full directional control",
            "category": "Toys",
            "age_range": "6-10",
        },
        {
            "title": "Art Set",
            "price": "$24.99",
            "description": "Complete art set with paints, brushes, and canvas",
            "category": "Arts & Crafts",
            "age_range": "5-15",
        },
        {
            "title": "Science Kit",
            "price": "$34.99",
            "description": "Educational science experiments kit",
            "category": "Educational",
            "age_range": "8-14",
        },
        {
            "title": "Dollhouse",
            "price": "$89.99",
            "description": "Beautiful wooden dollhouse with furniture",
            "category": "Toys",
            "age_range": "4-10",
        },
        {
            "title": "Building Blocks Set",
            "price": "$39.99",
            "description": "Classic wooden building blocks in various shapes and colors",
            "category": "Educational",
            "age_range": "3-8",
        },
        {
            "title": "Board Game Collection",
            "price": "$44.99",
            "description": "Set of 5 classic family board games",
            "category": "Games",
            "age_range": "6-99",
        },
        {
            "title": "Puppet Theater",
            "price": "$59.99",
            "description": "Wooden puppet theater with 6 hand puppets",
            "category": "Creative Play",
            "age_range": "4-12",
        },
        {
            "title": "Robot Building Kit",
            "price": "$69.99",
            "description": "Build and program your own robot with this STEM kit",
            "category": "Educational",
            "age_range": "10-16",
        },
        {
            "title": "Play Kitchen",
            "price": "$79.99",
            "description": "Realistic play kitchen with sounds and accessories",
            "category": "Pretend Play",
            "age_range": "3-8",
        },
        {
            "title": "Musical Instrument Set",
            "price": "$45.99",
            "description": "Collection of kid-friendly musical instruments",
            "category": "Music",
            "age_range": "3-10",
        },
        {
            "title": "Sports Equipment Pack",
            "price": "$54.99",
            "description": "Complete set of kids' sports gear including ball, bat, and net",
            "category": "Sports",
            "age_range": "6-12",
        },
        {
            "title": "Magic Tricks Kit",
            "price": "$29.99",
            "description": "Professional magic set with instruction manual",
            "category": "Entertainment",
            "age_range": "8-15",
        },
        {
            "title": "Dinosaur Collection",
            "price": "$39.99",
            "description": "Set of 12 detailed dinosaur figures with fact cards",
            "category": "Educational",
            "age_range": "4-12",
        },
        {
            "title": "Craft Supply Bundle",
            "price": "$49.99",
            "description": "Comprehensive craft supplies including beads, yarn, and tools",
            "category": "Arts & Crafts",
            "age_range": "6-16",
        },
        {
            "title": "Coding for Kids Set",
            "price": "$64.99",
            "description": "Interactive coding kit with programmable robot and game cards",
            "category": "STEM",
            "age_range": "8-14",
        },
        {
            "title": "Dress Up Trunk",
            "price": "$49.99",
            "description": "Collection of costumes and accessories for imaginative play",
            "category": "Pretend Play",
            "age_range": "3-10",
        },
        {
            "title": "Microscope Kit",
            "price": "$59.99",
            "description": "Real working microscope with prepared slides and tools",
            "category": "Science",
            "age_range": "10-15",
        },
        {
            "title": "Outdoor Explorer Kit",
            "price": "$34.99",
            "description": "Nature exploration set with binoculars, compass, and field guide",
            "category": "Outdoor",
            "age_range": "7-12",
        },
        {
            "title": "Art Pottery Studio",
            "price": "$69.99",
            "description": "Complete pottery wheel set with clay and glazing materials",
            "category": "Arts & Crafts",
            "age_range": "8-16",
        },
    ]
}

"""
## Initialize MongoDB Atlas

First, we need to set up our MongoDB Atlas collection and create a vector search index. This step is crucial for enabling semantic search capabilities:
"""
logger.info("## Initialize MongoDB Atlas")



client = MongoClient(
    os.environ["MONGO_CONNECTION_STRING"],
    appname="devrel.showcase.haystack_gifting_agent",
)
db = client["santa_workshop"]
collection = db["gifts"]

db.create_collection("gifts")


search_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "numDimensions": 1536,
                "path": "embedding",
                "similarity": "cosine",
            },
        ]
    },
    name="vector_index",
    type="vectorSearch",
)
result = collection.create_search_index(model=search_index_model)
logger.debug("New search index named " + result + " is building.")
logger.debug("Polling to check if the index is ready. This may take up to a minute.")
predicate = None
if predicate is None:
    predicate = lambda index: index.get("queryable") is True
while True:
    indices = list(collection.list_search_indexes(result))
    if len(indices) and predicate(indices[0]):
        break
    time.sleep(5)
logger.debug(result + " is ready for querying.")
client.close()

"""
## Initialize Document Store and Index Documents

Now let's set up the [MongoDBAtlasDocumentStore](https://docs.haystack.deepset.ai/docs/mongodbatlasdocumentstore) and index our gift data:
"""
logger.info("## Initialize Document Store and Index Documents")


document_store = MongoDBAtlasDocumentStore(
    database_name="santa_workshop",
    collection_name="gifts",
    vector_search_index="vector_index",
)

insert_data = []
for gift in dataset["train"]:
    doc_gift = json_util.loads(json_util.dumps(gift))
    haystack_doc = Document(content=doc_gift["title"], meta=doc_gift)
    insert_data.append(haystack_doc)

doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
doc_embedder = OllamaDocumentEmbedder(
    model="mxbai-embed-large", meta_fields_to_embed=["description"]
)

indexing_pipe = Pipeline()
indexing_pipe.add_component(instance=doc_embedder, name="doc_embedder")
indexing_pipe.add_component(instance=doc_writer, name="doc_writer")
indexing_pipe.connect("doc_embedder.documents", "doc_writer.documents")

indexing_pipe.run({"doc_embedder": {"documents": insert_data}})

"""
## Create Self-Reflecting Gift Selection Pipeline

Now comes the fun part! Create a pipeline that can:
1. Take a gift request query
2. Find relevant gifts using vector search
3. Self-reflect on selections to optimize for budget and preferences

You need a custom `GiftChecker` component that can if the more optimizateion is required. Learn how to write your Haystack component in [Docs: Creating Custom Components](https://docs.haystack.deepset.ai/docs/custom-components)
"""
logger.info("## Create Self-Reflecting Gift Selection Pipeline")




@component
class GiftChecker:
    @component.output_types(gifts_to_check=str, gifts=str)
    def run(self, replies: List[str]):
        if "DONE" in replies[0]:
            return {"gifts": replies[0].replace("DONE", "")}
        else:
            logger.debug(Fore.RED + "Not optimized yet, could find better gift combinations")
            return {"gifts_to_check": replies[0]}


prompt_template = """
    You are Santa's gift selection assistant . Below you have a list of available gifts with their prices.
    Based on the child's wishlist and budget, suggest appropriate gifts that maximize joy while staying within budget.

    Available Gifts:
    {% for doc in documents %}
        Gift: {{ doc.content }}
        Price: {{ doc.meta['price']}}
        Age Range: {{ doc.meta['age_range']}}
    {% endfor %}

    Query: {{query}}
    {% if gifts_to_check %}
        Previous gift selection: {{gifts_to_check[0]}}
        Can we optimize this selection for better value within budget?
        If optimal, say 'DONE' and return the selection
        If not, suggest a better combination
    {% endif %}

    Gift Selection:
"""

gift_pipeline = Pipeline(max_runs_per_component=5)
gift_pipeline.add_component(
    "text_embedder", OllamaTextEmbedder(model="mxbai-embed-large")
)
gift_pipeline.add_component(
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5),
    name="retriever",
)
gift_pipeline.add_component(
    instance=PromptBuilder(template=prompt_template), name="prompt_builder"
)
gift_pipeline.add_component(instance=GiftChecker(), name="checker")
gift_pipeline.add_component(instance=OllamaGenerator(model="llama3.2", log_dir=f"{LOG_DIR}/chats"), name="llm")

gift_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
gift_pipeline.connect("retriever.documents", "prompt_builder.documents")
gift_pipeline.connect("checker.gifts_to_check", "prompt_builder.gifts_to_check")
gift_pipeline.connect("prompt_builder", "llm")
gift_pipeline.connect("llm", "checker")

"""
## Test Your Gift Selection Agent

Let's test our pipeline with a sample query:
"""
logger.info("## Test Your Gift Selection Agent")

query = (
    "Find gifts for a 9-year-old who loves science and building things. Budget: $100"
)

result = gift_pipeline.run(
    {"text_embedder": {"text": query}, "prompt_builder": {"query": query}}
)

logger.debug(Fore.GREEN + result["checker"]["gifts"])

logger.info("\n\n[DONE]", bright=True)