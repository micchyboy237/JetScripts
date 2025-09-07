from bson import json_util
from colorama import Fore
from haystack import Document, Pipeline
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import OllamaDocumentEmbedder, OllamaTextEmbedder
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
from typing import List
import datetime
import json
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/MongoDB_Haystack_self_reflecting_Cooking_agent.ipynb)

# Haystack and MongoDB Atlas Agentic RAG pipelines

Haystack and MongoDB enhanced example building on top of the basic RAG pipeline demonstrated on the following [notebook](https://github.com/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/haystack_mongodb_cooking_advisor_pipeline.ipynb). Here the pipelines uses advanced technics of self reflection to advise on reciepes considering prices associated from the MongoDB Atlas vector store.

Install dependencies:
"""
logger.info("# Haystack and MongoDB Atlas Agentic RAG pipelines")

pip install haystack-ai mongodb-atlas-haystack tiktoken datasets

"""
## Setup MongoDB Atlas connection and Open AI


* Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI. If you wish to use google collab we recommend to allow access on Atlas Network tab to `0.0.0.0/0` so the notebook node can access the database.

* Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Setup MongoDB Atlas connection and Open AI")

# import getpass

# os.environ["MONGO_CONNECTION_STRING"] = getpass.getpass(
    "Enter your MongoDB connection string:"
)

# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Open AI Key:")

"""
## Create vector search index on collection

Follow this [tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/) to create a vector index on database: `haystack_test` collection `test_collection`.

Verify that the index name is `vector_index` and the syntax specify:
```
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

### Setup vector store to load documents:
"""
logger.info("## Create vector search index on collection")


dataset = {
    "train": [
        {
            "title": "Spinach Lasagna Sheets",
            "price": "$3.50",
            "description": "Infused with spinach, these sheets add a pop of color and extra nutrients.",
            "category": "Pasta",
            "emoji": "📗",
        },
        {
            "title": "Gluten-Free Lasagna Sheets",
            "price": "$4.00",
            "description": "Perfect for those with gluten intolerance, made with a blend of rice and corn flour.",
            "category": "Pasta",
            "emoji": "🍚🌽",
        },
        {
            "title": "Mascarpone",
            "price": "$4.00",
            "description": "Creamy and rich, this cheese adds a luxurious touch to lasagna.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Queso Fresco",
            "price": "$3.00",
            "description": "A mild, crumbly cheese that can be a suitable replacement for ricotta.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Vegetarian Lentil Sauce",
            "price": "$4.00",
            "description": "A meatless option made with cooked lentils that mimics the texture of ground meat.",
            "category": "Vegetarian",
            "emoji": "🍲",
        },
        {
            "title": "Turkey Bolognese",
            "price": "$5.00",
            "description": "A leaner alternative to beef, turkey provides a lighter but flavorful taste.",
            "category": "Poultry",
            "emoji": "🦃",
        },
        {
            "title": "Mushroom and Walnut Sauce",
            "price": "$5.50",
            "description": "Combining chopped mushrooms and walnuts for a hearty vegetarian filling.",
            "category": "Vegetarian",
            "emoji": "🍄🥜",
        },
        {
            "title": "Chicken Bolognese",
            "price": "$5.00",
            "description": "Ground chicken offers a different twist on the classic meat sauce.",
            "category": "Poultry",
            "emoji": "🐔",
        },
        {
            "title": "Vegan Soy Meat Sauce",
            "price": "$4.50",
            "description": "Made from soy protein, this vegan meat sauce replicates the texture and flavor of traditional meat.",
            "category": "Vegan",
            "emoji": "🌱",
        },
        {
            "title": "Tomato Basil Sauce",
            "price": "$3.50",
            "description": "A tangy alternative to béchamel, made with fresh tomatoes and basil.",
            "category": "Vegetarian",
            "emoji": "🍅",
        },
        {
            "title": "Pesto Cream Sauce",
            "price": "$5.50",
            "description": "A fusion of creamy béchamel and rich basil pesto for a unique flavor.",
            "category": "Dairy",
            "emoji": "🍝",
        },
        {
            "title": "Alfredo Sauce",
            "price": "$4.50",
            "description": "A rich and creamy white sauce made with parmesan and butter.",
            "category": "Dairy",
            "emoji": "🧈",
        },
        {
            "title": "Coconut Milk Béchamel",
            "price": "$4.00",
            "description": "A dairy-free version of the classic béchamel made with coconut milk.",
            "category": "Vegan",
            "emoji": "🥥",
        },
        {
            "title": "Vegan Cashew Cream Sauce",
            "price": "$5.00",
            "description": "A rich and creamy sauce made from blended cashews as a dairy-free alternative.",
            "category": "Vegan",
            "emoji": "🥜",
        },
        {
            "title": "Kale",
            "price": "$2.00",
            "description": "Another leafy green option, kale offers a chewy texture and rich nutrients.",
            "category": "Leafy Greens",
            "emoji": "🥬",
        },
        {
            "title": "Bell Peppers",
            "price": "$2.50",
            "description": "Sliced bell peppers in various colors add sweetness and crunch.",
            "category": "Vegetables",
            "emoji": "🫑",
        },
        {
            "title": "Artichoke Hearts",
            "price": "$3.50",
            "description": "Tender and flavorful, artichoke hearts bring a Mediterranean twist to the dish.",
            "category": "Vegetables",
            "emoji": "🍽️",
        },
        {
            "title": "Spinach",
            "price": "$2.00",
            "description": "Fresh or frozen spinach adds a pop of color and nutrients.",
            "category": "Leafy Greens",
            "emoji": "🥬",
        },
        {
            "title": "Broccoli",
            "price": "$2.50",
            "description": "Small broccoli florets provide texture and a distinct flavor.",
            "category": "Vegetables",
            "emoji": "🥦",
        },
        {
            "title": "Whole Wheat Lasagna Sheets",
            "price": "$3.00",
            "description": "Made from whole wheat grains, these sheets are healthier and provide a nutty flavor.",
            "category": "Pasta",
            "emoji": "🌾",
        },
        {
            "title": "Zucchini Slices",
            "price": "$2.50",
            "description": "Thinly sliced zucchini can replace traditional pasta for a low-carb version.",
            "category": "Vegetables",
            "emoji": "🥒",
        },
        {
            "title": "Eggplant Slices",
            "price": "$2.75",
            "description": "Thin slices of eggplant provide a meaty texture, ideal for vegetarian lasagna.",
            "category": "Vegetables",
            "emoji": "🍆",
        },
        {
            "title": "Ground Turkey",
            "price": "$4.50",
            "description": "A leaner alternative to beef, turkey provides a lighter but flavorful taste.",
            "category": "Meat",
            "emoji": "🦃",
        },
        {
            "title": "Vegetarian Lentil Mince",
            "price": "$3.50",
            "description": "A meatless option made with cooked lentils that mimics the texture of ground meat.",
            "category": "Vegetarian",
            "emoji": "🍲",
        },
        {
            "title": "Mushroom and Walnut Mince",
            "price": "$5.00",
            "description": "Combining chopped mushrooms and walnuts for a hearty vegetarian filling.",
            "category": "Vegetarian",
            "emoji": "🍄🥜",
        },
        {
            "title": "Ground Chicken",
            "price": "$4.00",
            "description": "Ground chicken offers a different twist on the classic meat sauce.",
            "category": "Poultry",
            "emoji": "🐔",
        },
        {
            "title": "Vegan Soy Meat Crumbles",
            "price": "$4.50",
            "description": "Made from soy protein, these crumbles replicate the texture and flavor of traditional meat.",
            "category": "Vegan",
            "emoji": "🥩",
        },
        {
            "title": "Pesto Sauce",
            "price": "$4.00",
            "description": "A green, aromatic sauce made from basil, pine nuts, and garlic.",
            "category": "Canned Goods",
            "emoji": "🌿",
        },
        {
            "title": "Marinara Sauce",
            "price": "$3.50",
            "description": "A classic Italian tomato sauce with garlic, onions, and herbs.",
            "category": "Canned Goods",
            "emoji": "🍅",
        },
        {
            "title": "Bolognese Sauce",
            "price": "$5.00",
            "description": "A meat-based sauce simmered with tomatoes, onions, celery, and carrots.",
            "category": "Canned Goods",
            "emoji": "🍖🍅🧅🥕",
        },
        {
            "title": "Arrabbiata Sauce",
            "price": "$4.00",
            "description": "A spicy tomato sauce made with red chili peppers.",
            "category": "Canned Goods",
            "emoji": "🌶️🍅",
        },
        {
            "title": "Provolone Cheese",
            "price": "$3.50",
            "description": "Semi-hard cheese with a smooth texture, it melts beautifully in dishes.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Cheddar Cheese",
            "price": "$3.00",
            "description": "A popular cheese with a sharp and tangy flavor profile.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Gouda Cheese",
            "price": "$4.50",
            "description": "A Dutch cheese known for its rich and creamy texture.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Fontina Cheese",
            "price": "$4.00",
            "description": "A semi-soft cheese with a strong flavor, great for melting.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Vegan Mozzarella",
            "price": "$5.00",
            "description": "Dairy-free alternative made from nuts or soy, melts similarly to regular mozzarella.",
            "category": "Vegan",
            "emoji": "🧀",
        },
        {
            "title": "Cottage Cheese",
            "price": "$2.50",
            "description": "A lighter alternative to ricotta, with small curds that provide a similar texture.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Goat Cheese",
            "price": "$4.00",
            "description": "A tangy and creamy cheese that can provide a unique flavor to lasagna.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Mascarpone Cheese",
            "price": "$4.50",
            "description": "An Italian cream cheese with a rich and creamy texture.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Tofu Ricotta",
            "price": "$3.00",
            "description": "A vegan alternative made from crumbled tofu seasoned with herbs.",
            "category": "Vegan",
            "emoji": "🌱",
        },
        {
            "title": "Feta Cheese",
            "price": "$3.50",
            "description": "A crumbly cheese with a salty profile, it can bring a Mediterranean twist to the dish.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Parmesan cheese",
            "price": "$4.00",
            "description": "A hard, granular cheese originating from Italy, known for its rich umami flavor.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Pecorino Romano",
            "price": "$5.00",
            "description": "A salty, hard cheese made from sheep's milk, perfect for grating over dishes.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Asiago Cheese",
            "price": "$4.50",
            "description": "Semi-hard cheese with a nutty flavor, great for shaving or grating.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Grana Padano",
            "price": "$5.50",
            "description": "A grainy, hard cheese that's similar to Parmesan but milder in flavor.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Manchego Cheese",
            "price": "$6.00",
            "description": "A Spanish hard cheese with a rich and nutty flavor.",
            "category": "Dairy",
            "emoji": "🧀",
        },
        {
            "title": "Eggs",
            "price": "$2.00",
            "description": "Rich in protein and versatile, eggs are used in a variety of culinary applications.",
            "category": "Dairy",
            "emoji": "🥚",
        },
        {
            "title": "Tofu",
            "price": "$2.00",
            "description": "Blended silken tofu can act as a binder in various dishes.",
            "category": "Vegan",
            "emoji": "🍲",
        },
        {
            "title": "Flaxseed Meal",
            "price": "$1.50",
            "description": "Mix with water to create a gel-like consistency that can replace eggs.",
            "category": "Vegan",
            "emoji": "🥚",
        },
        {
            "title": "Chia Seeds",
            "price": "$2.50",
            "description": "Mix with water to form a gel that can be used as an egg substitute.",
            "category": "Vegan",
            "emoji": "🥚",
        },
        {
            "title": "Apple Sauce",
            "price": "$2.00",
            "description": "A sweet alternative that can replace eggs in certain recipes.",
            "category": "Baking",
            "emoji": "🥚",
        },
        {
            "title": "Onion",
            "price": "$1.00",
            "description": "A kitchen staple, onions provide depth and flavor to a myriad of dishes.",
            "category": "Vegetables",
            "emoji": "🧅",
        },
        {
            "title": "Shallots",
            "price": "$2.00",
            "description": "Milder and sweeter than regular onions, they add a delicate flavor.",
            "category": "Produce",
            "emoji": "🧅",
        },
        {
            "title": "Green Onions",
            "price": "$1.50",
            "description": "Milder in flavor, green onions or scallions are great for garnishing.",
            "category": "Vegetables",
            "emoji": "🌱",
        },
        {
            "title": "Red Onion",
            "price": "$1.20",
            "description": "Sweeter and more vibrant in color, red onions add a pop to dishes.",
            "category": "Vegetables",
            "emoji": "🔴",
        },
        {
            "title": "Leeks",
            "price": "$2.50",
            "description": "With a light onion flavor, leeks are great when sautéed or used in soups.",
            "category": "Produce",
            "emoji": "🍲",
        },
        {
            "title": "Garlic",
            "price": "$0.50",
            "description": "Aromatic and flavorful, garlic is a foundational ingredient in many cuisines.",
            "category": "Produce",
            "emoji": "🧄",
        },
        {
            "title": "Garlic Powder",
            "price": "$2.00",
            "description": "A convenient dried version of garlic that provides a milder flavor.",
            "category": "Spices",
            "emoji": "🧄",
        },
        {
            "title": "Garlic Flakes",
            "price": "$2.50",
            "description": "Dried garlic flakes can be rehydrated or used as they are for a burst of garlic flavor.",
            "category": "Spices",
            "emoji": "🧄",
        },
        {
            "title": "Garlic Paste",
            "price": "$3.00",
            "description": "A smooth blend of garlic, perfect for adding to sauces or marinades.",
            "category": "Condiments",
            "emoji": "🧄",
        },
        {
            "title": "Olive Oil",
            "price": "$6.00",
            "description": "A staple in Mediterranean cuisine, olive oil is known for its heart-healthy properties.",
            "category": "Condiments",
            "emoji": "🍽️",
        },
        {
            "title": "Canola Oil",
            "price": "$3.50",
            "description": "A neutral-tasting oil suitable for various cooking methods.",
            "category": "Condiments",
            "emoji": "🍳",
        },
        {
            "title": "Coconut Oil",
            "price": "$5.00",
            "description": "A fragrant oil ideal for sautéing and baking.",
            "category": "Condiments",
            "emoji": "🍳",
        },
        {
            "title": "Avocado Oil",
            "price": "$7.00",
            "description": "Known for its high smoke point, it's great for high-heat cooking.",
            "category": "Condiments",
            "emoji": "🍳",
        },
        {
            "title": "Grapeseed Oil",
            "price": "$6.50",
            "description": "A light, neutral oil that's good for dressings and sautéing.",
            "category": "Condiments",
            "emoji": "🥗",
        },
        {
            "title": "Salt",
            "price": "$1.00",
            "description": "An essential seasoning that enhances the flavor of dishes.",
            "category": "Spices",
            "emoji": "🧂",
        },
        {
            "title": "Himalayan Pink Salt",
            "price": "$2.50",
            "description": "A natural and unrefined salt with a slightly earthy flavor.",
            "category": "Spices",
            "emoji": "🧂",
        },
        {
            "title": "Sea Salt",
            "price": "$2.00",
            "description": "Derived from evaporated seawater, it provides a briny touch.",
            "category": "Spices",
            "emoji": "🌊",
        },
        {
            "title": "Kosher Salt",
            "price": "$1.50",
            "description": "A coarse salt without additives, commonly used in cooking.",
            "category": "Spices",
            "emoji": "🧂",
        },
        {
            "title": "Black Salt (Kala Namak)",
            "price": "$2.00",
            "description": "A sulfurous salt often used in South Asian cuisine, especially vegan dishes to mimic an eggy flavor.",
            "category": "Spices",
            "emoji": "🧂",
        },
        {
            "title": "Black Pepper",
            "price": "$2.00",
            "description": "A versatile spice known for its sharp and mildly spicy flavor.",
            "category": "Spices",
            "emoji": "🌶️",
        },
        {
            "title": "White Pepper",
            "price": "$2.50",
            "description": "Milder than black pepper, it's often used in light-colored dishes.",
            "category": "Spices",
            "emoji": "🌶️",
        },
        {
            "title": "Cayenne Pepper",
            "price": "$2.00",
            "description": "A spicy chili pepper, ground into powder. Adds heat to dishes.",
            "category": "Spices",
            "emoji": "🌶️",
        },
        {
            "title": "Crushed Red Pepper Flakes",
            "price": "$1.50",
            "description": "Adds a spicy kick to dishes, commonly used as a pizza topping.",
            "category": "Spices",
            "emoji": "🌶️",
        },
        {
            "title": "Sichuan (or Szechuan) Peppercorns",
            "price": "$3.00",
            "description": "Known for their unique tingling sensation, they're used in Chinese cuisine.",
            "category": "Spices",
            "emoji": "🥡",
        },
        {
            "title": "Banana",
            "price": "$0.60",
            "description": "A sweet and portable fruit, packed with essential vitamins.",
            "category": "Produce",
            "emoji": "🍌",
        },
        {
            "title": "Milk",
            "price": "$2.50",
            "description": "A calcium-rich dairy product, perfect for drinking or cooking.",
            "category": "Dairy",
            "emoji": "🥛",
        },
        {
            "title": "Bread",
            "price": "$2.00",
            "description": "Freshly baked, perfect for sandwiches or toast.",
            "category": "Bakery",
            "emoji": "🍞",
        },
        {
            "title": "Apple",
            "price": "$1.00",
            "description": "Crisp and juicy, great for snacking or baking.",
            "category": "Produce",
            "emoji": "🍏",
        },
        {
            "title": "Orange",
            "price": "3.99$",
            "description": "Great as a juice and vitamin",
            "category": "Produce",
            "emoji": "🍊",
        },
        {
            "title": "Sugar",
            "price": "1.00",
            "description": "very sweet substance",
            "category": "Spices",
            "emoji": "🍰",
        },
    ]
}

insert_data = []

for product in dataset["train"]:
    doc_product = json_util.loads(json_util.dumps(product))
    haystack_doc = Document(content=doc_product["title"], meta=doc_product)
    insert_data.append(haystack_doc)


document_store = MongoDBAtlasDocumentStore(
    database_name="ai_shop",
    collection_name="test_collection",
    vector_search_index="vector_index",
)

"""
Build the writer pipeline to load documnets
"""
logger.info("Build the writer pipeline to load documnets")

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
## Build a Pipeline to have

First lets add prices to the augmenting considerations by enhancing our prompt template with Price: `{{ doc.meta['price']}}`
"""
logger.info("## Build a Pipeline to have")

prompt_template = """
    You are a recipe builder assistant. Below you have a list of ingredients followed by its price for each ingredient.
    Based on the requested food, provide a step by step recipe, followed by an itemized and total shopping list cost.

    Your recipe should have the following sections:
    - Ingredients
    - Steps
    - Cost

    {% for doc in documents %}
        Ingredient: {{ doc.content }}
        Price: {{ doc.meta['price']}}
    {% endfor %}

    Query: {{query}}

    Recipe:
"""

rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "text_embedder", OllamaTextEmbedder(model="mxbai-embed-large")
)

rag_pipeline.add_component(
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=50),
    name="retriever",
)

rag_pipeline.add_component(
    instance=PromptBuilder(template=prompt_template), name="prompt_builder"
)

rag_pipeline.add_component(instance=OllamaGenerator(model="llama3.2", log_dir=f"{LOG_DIR}/chats"), name="llm")

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

query = "How can I cook a lasagne?"
result = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query},
    },
    include_outputs_from=["prompt_builder"],
)
logger.debug(result["llm"]["replies"][0])

"""
## Make it cheaper with self-reflection!

Here the agentic workflow is built around self reflection of the LLM to reconsider the suggested set of ingridiants in order to find the cheapest reciepe possible.
"""
logger.info("## Make it cheaper with self-reflection!")

# !pip install colorama




@component
class RecipeChecker:
    @component.output_types(recipe_to_check=str, recipe=str)
    def run(self, replies: List[str]):
        if "DONE" in replies[0]:
            return {"recipe": replies[0].replace("done", "")}
        logger.debug(Fore.RED + "Not done yet, could make recipe more efficient")
        return {"recipe_to_check": replies[0]}

prompt_template = """
    You are a recipe builder assistant. Below you have a list of ingredients followed by its price for each ingredient.
    Based on the requested food, provide a step by step recipe, followed by an itemized and total shopping list cost.

    Your recipe should have the following sections:
    - Ingredients
    - Steps
    - Cost

    {% for doc in documents %}
        Ingredient: {{ doc.content }}
        Price: {{ doc.meta['price']}}
    {% endfor %}

    Query: {{query}}
    {% if recipe_to_check %}
        Here is the recipe you previously generated: {{recipe_to_check[0]}}
        Is this the most efficient and cheap way to do this recipe?
        If yes, say 'DONE' and return the recipe s in the next line
        If not, say 'incomplete' and return the recipe in the next line
    {% endif %}
    \nRecipe:
"""

reflecting_rag_pipeline = Pipeline(max_loops_allowed=5)
reflecting_rag_pipeline.add_component(
    "text_embedder", OllamaTextEmbedder(model="mxbai-embed-large")
)
reflecting_rag_pipeline.add_component(
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=50),
    name="retriever",
)
reflecting_rag_pipeline.add_component(
    instance=PromptBuilder(template=prompt_template), name="prompt_builder"
)
reflecting_rag_pipeline.add_component(instance=RecipeChecker(), name="checker")
reflecting_rag_pipeline.add_component(
    instance=OllamaGenerator(model="llama3.2", log_dir=f"{LOG_DIR}/chats"), name="llm"
)

reflecting_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
reflecting_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
reflecting_rag_pipeline.connect(
    "checker.recipe_to_check", "prompt_builder.recipe_to_check"
)
reflecting_rag_pipeline.connect("prompt_builder", "llm")
reflecting_rag_pipeline.connect("llm", "checker")

reflecting_rag_pipeline.show()

"""
As you can see the pipeline will loop through itself to find a more efficient reciepe.
"""
logger.info("As you can see the pipeline will loop through itself to find a more efficient reciepe.")

query = "How can I cook a lasagne?"
result = reflecting_rag_pipeline.run(
    {"text_embedder": {"text": query}, "prompt_builder": {"query": query}}
)
logger.debug(Fore.GREEN + result["checker"]["recipe"])

"""
## Use JSON format output

Developers will usually prefer dealing with a JSON format output from LLMs when building applications, as well the ease of storing JSON objects in MongoDB Atlas for fututre store and use.
"""
logger.info("## Use JSON format output")

prompt_template = """
    You are a recipe builder assistant. Below you have a list of ingredients followed by its price for each ingredient.
    Respond in JSON format to include only relevant reciepe data, it must have all the markdown under 'markdown_text' field, checker_status : ..., 'ingridiants' : []
    Based on the requested food, provide a step by step recipe, followed by an itemized and total shopping list cost.

    Your recipe should have the following sections:
    - Ingredients
    - Steps
    - Cost

    {% for doc in documents %}
        Ingredient: {{ doc.content }}
        Price: {{ doc.meta['price']}}
    {% endfor %}

    Query: {{query}}
    {% if recipe_to_check %}
        Here is the recipe you previously generated: {{recipe_to_check[0]}}
        Is this the most efficient and cheap way to do this recipe?
        If yes, say 'checker_status' : 'DONE' and return the recipe s in the next line
        If not, say 'incomplete' and return the recipe in the next line
    {% endif %}
    \nRecipe:
"""

reflecting_rag_pipeline = Pipeline(max_loops_allowed=10)
reflecting_rag_pipeline.add_component(
    "text_embedder", OllamaTextEmbedder(model="mxbai-embed-large")
)
reflecting_rag_pipeline.add_component(
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=50),
    name="retriever",
)
reflecting_rag_pipeline.add_component(
    instance=PromptBuilder(template=prompt_template), name="prompt_builder"
)
reflecting_rag_pipeline.add_component(instance=RecipeChecker(), name="checker")
reflecting_rag_pipeline.add_component(
    instance=OllamaGenerator(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats",
        generation_kwargs={
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    ),
    name="llm",
)

reflecting_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
reflecting_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
reflecting_rag_pipeline.connect(
    "checker.recipe_to_check", "prompt_builder.recipe_to_check"
)
reflecting_rag_pipeline.connect("prompt_builder", "llm")
reflecting_rag_pipeline.connect("llm", "checker")

# !pip install pymongo



query = "How can I cook a lasagne?"
result = reflecting_rag_pipeline.run(
    {"text_embedder": {"text": query}, "prompt_builder": {"query": query}}
)
logger.debug(Fore.GREEN + result["checker"]["recipe"])

doc = json.loads(result["checker"]["recipe"])

doc["date"] = datetime.datetime.now()

mongo_client = MongoClient(
    os.environ["MONGO_CONNECTION_STRING"],
    appname="devrel.showcase.haystack_cooking_agent",
)
db = mongo_client["ai_shop"]
collection = db["reciepes"]
collection.insert_one(doc)

logger.info("\n\n[DONE]", bright=True)