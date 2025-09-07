from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain_core.documents import Document
from langchain_core.tools import tool  # Import tool decorator
from langchain_mongodb.vectorstores import (
MongoDBAtlasVectorSearch,
)  # Import necessary class
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.mongodb.base import (
MongoDBStore,
VectorIndexConfig,
)  # Import MongoDBStore
from langmem import create_manage_memory_tool
from ollama import Ollama
from pymongo import MongoClient
from typing_extensions import NotRequired
import os
import shutil
import sys
import typing


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Langmem with MongoDB: Building Conversational Memory

---

This notebook demonstrates how to use `langmem` with `langgraph` and MongoDB to build a conversational agent that can remember information across interactions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/memory/langmem_mongodb_notebook.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://dev.to/mongodb/langgraph-with-mongodb-building-conversational-long-term-memory-for-intelligent-ai-agents-2pcn)

## Use Case: A Simple Preference-Aware Assistant

We will build a simple assistant that can remember a user's preferences (like their preferred display mode) and recall them in the same or different conversations.

### Part 1: Setup and Dependencies

First, let's install the required Python libraries.
"""
logger.info("# Langmem with MongoDB: Building Conversational Memory")

# !pip install -qU  langgraph langmem pymongo ollama python-dotenv typing-extensions langgraph-checkpoint-mongodb langchain_mongodb langgraph-store-mongodb langchain_voyageai

"""
### Setup API Keys
"""
logger.info("### Setup API Keys")

# import getpass

# OPENAI_API_KEY = getpass.getpass("Ollama API Key: ")
# MONGODB_URI = getpass.getpass("MongoDB URI: ")
# VOYAGE_API_KEY = getpass.getpass("VoyageAI API Key: ")

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["MONGODB_URI"] = MONGODB_URI
os.environ["VOYAGE_API_KEY"] = VOYAGE_API_KEY

"""
### Step 2: Importing Libraries and Setting Up Environment

Now, we'll import the necessary modules and set up our API keys for Ollama and MongoDB. We'll use `python-dotenv` to load credentials from a `.env` file.
"""
logger.info("### Step 2: Importing Libraries and Setting Up Environment")




if sys.version_info < (3, 11):

    typing.NotRequired = NotRequired

load_dotenv()


# client = Ollama(api_key=os.getenv("OPENAI_API_KEY"))
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set.")

"""
### Loading Data set

We will use the "ai_shop" catalog dataset which has a product catalog with title, descripition and pricing data.
"""
logger.info("### Loading Data set")

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

"""
### Load the data and create vector store index.

We will use a langchain vector store adapter to load and embed our catalog. This will allow the agent to search for relevant products from our database.
"""
logger.info("### Load the data and create vector store index.")


docs = [
    Document(page_content=f"{item['title']} - {item['description']}", metadata=item)
    for item in dataset["train"]
]

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace="ai_shop.products",
    embedding=VoyageAIEmbeddings(model="voyage-3.5"),
    index_name="vector_index",
)

client = MongoClient(MONGODB_URI)
db = client["ai_shop"]
collection = db["products"]

collection.delete_many({})

vector_store.add_documents(documents=docs)

vector_store.create_vector_search_index(dimensions=1024, wait_until_complete=70)

logger.debug("Database seeded successfully!")

"""
### Part 2: Core Components Explained

Our agent has a few key parts:

- **`MongoDBStore`**: This is where the agent's "memories" are stored for fast retrieval using vector search. It's great for semantic search with persistant nature.
- **`MongoDBSaver`**: This acts as a checkpointer. It saves the entire conversation state (including messages) to a MongoDB database, ensuring durability and allowing conversations to be resumed.
- **`create_manage_memory_tool`**: This tool, provided by `langmem`, gives the agent the ability to create, update, and delete memories in its `MongoDBStore`.
- **`prompt` function**: This function dynamically injects relevant memories from the `MongoDBStore` into the agent's system prompt, giving it context for its responses.
"""
logger.info("### Part 2: Core Components Explained")

def prompt(state, store):
    """Prepare the messages for the LLM by injecting memories."""
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a shopping assistant that have access to search_products tool and memory


<memories>
{memories}
</memories>

"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]

"""
### Part 3: Building and Running the Agent
We will leverge [`MongoDBStore`](https://langchain-mongodb.readthedocs.io/en/latest/langgraph_store_mongodb/base/langgraph.store.mongodb.base.MongoDBStore.html#langgraph.store.mongodb.base.MongoDBStore) to form a long term memory, this component will leverge semantic (vector) search to remember and fetch memories.
"""
logger.info("### Part 3: Building and Running the Agent")



client = MongoClient(MONGODB_URI)
db = client["memories"]
collection = db["memory_store"]

store = MongoDBStore(
    collection=collection,
    index_config=VectorIndexConfig(
        fields=None,
        filters=None,
        dims=1024,
        embed=VoyageAIEmbeddings(
            model="voyage-3.5"
        ),  # Pass an instance of OllamaEmbeddings
    ),
    auto_index_timeout=70,
)


@tool
def search_products(query: str) -> str:
    """Searches for products in the database using vector search."""
    db = client["ai_shop"]
    collection = db["products"]
    vectorstore = MongoDBAtlasVectorSearch(
        collection,
        VoyageAIEmbeddings(model="voyage-3.5"),
        text_key="title",
        embedding_key="embedding",
        index_name="vector_index",
    )
    docs = vectorstore.similarity_search(query, k=5)

    return "\n".join([str(doc.metadata) for doc in docs])


checkpointer = MongoDBSaver(
    client, db_name="memories", collection_name="thread_checkpoints"
)

agent = create_react_agent(
    "ollama:gpt-4o",
    prompt=lambda state: prompt(state, store),  # Pass the store to the prompt function
    tools=[
        create_manage_memory_tool(namespace=("memories",)),
        search_products,  # Add the new tool here
    ],
    store=store,
    checkpointer=checkpointer,
)

"""
#### Demonstration

Let's start a conversation. We use a `thread_id` to manage conversation state. The agent has no memories of us yet.
"""
logger.info("#### Demonstration")

config = {"configurable": {"thread_id": "thread-a"}}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my diet preference? I need help on finding ingiridiants for a lasagne, search the database for each ingridiant seperatly and provide overall total",
            }
        ]
    },
    config=config,
)
logger.debug(response["messages"][-1].content)

"""
Now, let's tell the agent our preference. The agent will use the `manage_memory` tool to save this information.
"""
logger.info("Now, let's tell the agent our preference. The agent will use the `manage_memory` tool to save this information.")

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Remember I am vegan. Please recalculate"}
        ]
    },
    config=config,
)

logger.debug(response["messages"][-1].content)

"""
If we ask again in the same conversation thread, it should remember.
"""
logger.info("If we ask again in the same conversation thread, it should remember.")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "So how is this aligned with my diet?"}]},
    config=config,
)
logger.debug(response["messages"][-1].content)

"""
Now, let's start a new conversation with a different `thread_id`. The agent will remeber us in this new thread, because the memories are scoped not scoped to a conversation.
"""
logger.info("Now, let's start a new conversation with a different `thread_id`. The agent will remeber us in this new thread, because the memories are scoped not scoped to a conversation.")

new_config = {"configurable": {"thread_id": "thread-b"}}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hey there. Do you remember me? What is my diet preference?",
            }
        ]
    },
    config=new_config,
)
logger.debug(response["messages"][-1].content)

"""
### ⛳ Knowledge Checkpoint

You've now seen how to:

- **Create a stateful agent** using `langgraph`.
- **Use `langmem`** to provide the agent with tools to manage its own memory.
- **Leverage MongoDB** with `MongoDBSaver` and `MongoDBStore` to persist conversation history and long-term memory, making your agent durable with memory mechanisim.
"""
logger.info("### ⛳ Knowledge Checkpoint")

logger.info("\n\n[DONE]", bright=True)