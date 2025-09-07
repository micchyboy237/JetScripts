async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from playwright.async_api import async_playwright
    from pymongo import MongoClient
    import json
    import ollama
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
    <a href="https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/SwigMenu_Playwright_Ollama_MongoDB.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    [![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/playwright-structured-outputs-atlas-search/)
    
    ## Overview
    
    In this tutorial we are going to scrape the popular Utah "dirty" soda website, Swig, using Playwright, then we are going to feed in our drinks into Ollama using a prompt and their structured outputs to understand which drinks from their menu are best for various seasons with reasonings, and then save this information into MongoDB Atlas so we can use Atlas Search to find specific drinks based on the fall season and ingredients we are craving.
    
    ## Part 1: Scrape all menu items from Swig website
    
    Let's first scrape all our menu items from the Swig website. We need to install Playwright and then build out our function.
    """
    logger.info("## Overview")
    
    # !pip install playwright
    # !playwright install
    
    """
    We have to use async since we are using Google Colab. If you're
    not using a notebook you can use sync instead. Please refer to the article written to understand where our selectors came from.
    """
    logger.info("We have to use async since we are using Google Colab. If you're")
    
    
    """
    We are using the URL that is inside of the websites iframe, and we are using selectors to make sure we are waiting for the information we want to load. We want to grab the name of each menu item along with its description. Please refer to the written article to understand this function better if necessary!
    """
    logger.info("We are using the URL that is inside of the websites iframe, and we are using selectors to make sure we are waiting for the information we want to load. We want to grab the name of each menu item along with its description. Please refer to the written article to understand this function better if necessary!")
    
    async def swigScraper():
        async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                page = await browser.new_page()
            
                await page.goto("https://swig-orders.crispnow.com/tabs/locations/menu")
            
                await page.wait_for_selector(
                    "ion-card-content", state="attached", timeout=60000
                )
            
                items = await page.query_selector_all("ion-card-content")
            
                menu = []
            
                for item in items:
                    result = {}
            
                    name = await item.query_selector("p.text-h3")
                    description = await item.query_selector("p.text-b2")
            
                    if name and description:
                        result = {}
                        result["name"] = await name.inner_text()
                        result["description"] = await description.inner_text()
                        menu.append(result)
            
                for item in menu:
                    logger.debug(f"Name: {item['name']}, Description: {item['description']}")
            
                await browser.close()
                return menu
            
            
        logger.success(format_json(result))
    scraped_menu = await swigScraper()
    logger.success(format_json(scraped_menu))
    
    logger.debug(scraped_menu)
    
    """
    Now that we have all of our menu options, let's use Ollama to tell us which drinks are best for fall based on their descriptions.
    
    ## Step 2: Ollama Structured Schema Outputs
    Please refer to the documentation to understand Ollama's structured schema outputs. We want to emulate the section where they are extracting structured data from unstructured data.
    """
    logger.info("## Step 2: Ollama Structured Schema Outputs")
    
    # !pip install ollama
    
    # import getpass
    
    
    # ollama_api_key = getpass.getpass(prompt="Put in Ollama API Key here")
    
    """
    Here we are formatting our menu from when we scraped it, putting everything into a single string for Ollama to understand, and then creating a prompt helping our model understand what we are hoping to achieve.
    """
    logger.info("Here we are formatting our menu from when we scraped it, putting everything into a single string for Ollama to understand, and then creating a prompt helping our model understand what we are hoping to achieve.")
    
    def swigJoined(scraped_menu):
        drink_list = []
    
        for drink in scraped_menu:
            drink_format = f"{drink['name']}: {drink['description']}]"
            drink_list.append(drink_format)
    
        drink_string = "\n".join(drink_list)
    
        prompt = (
            "You are the best soda mixologist Utah has ever seen! This is a list of sodas and their descriptions, or ingredients:\n"
            f"{drink_string}\n\n Please sort each and every drink provided into spring, summer, fall, or winter seasons based on their ingredients\n"
            "and give me reasonings as to why by stating which ingredients make it best for each season. For example, cinnamon is more fall, but peach\n"
            "is more summer."
        )
    
        return prompt
    
    my_prompt = swigJoined(scraped_menu)
    
    ollama.api_key = ollama_api_key
    
    response = ollama.chat.completions.create(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats",
        messages=[
            {
                "role": "system",
                "content": "You are the best soda mixologist Utah has ever seen!",
            },
            {"role": "user", "content": my_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "drink_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "seasonal_drinks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "drink": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["drink", "reason"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["seasonal_drinks"],
                    "additionalProperties": False,
                },
            },
        },
    )
    
    """
    Let's check and see our full response and see if it's structured the way we want.
    """
    logger.info("Let's check and see our full response and see if it's structured the way we want.")
    
    logger.debug(json.dumps(response.model_dump(), indent=2))
    
    """
    It is structured nicely, but all our fall drinks with their reasonings are under the "content" part. Let's open this up so we can better read it.
    """
    logger.info("It is structured nicely, but all our fall drinks with their reasonings are under the "content" part. Let's open this up so we can better read it.")
    
    content = response.model_dump()["choices"][0]["message"]["content"]
    logger.debug(content)
    
    """
    So it's still in one line. Let's print them out nicely for better readability and so when we input it into MongoDB Atlas everything is in different documents.
    """
    logger.info("So it's still in one line. Let's print them out nicely for better readability and so when we input it into MongoDB Atlas everything is in different documents.")
    
    parsed_drinks = json.loads(content)
    seasonal_drinks_pretty = parsed_drinks["seasonal_drinks"]
    logger.debug(json.dumps(seasonal_drinks_pretty, indent=2))
    
    """
    Now that our drinks with their reasonings are printed out nicely, let's upload them into MongoDB Atlas so we can use Atlas Search and take a look at drinks based off their ingredients!
    
    ## Step 3: Store into MongoDB and use Atlas Search
    
    For this section a MongoDB Atlas cluster is required. Please make sure you have your connection string saved somewhere safe.
    
    First install PyMongo to make things easier for ourselves.
    """
    logger.info("## Step 3: Store into MongoDB and use Atlas Search")
    
    # !pip install pymongo
    
    """
    Set up our MongoDB connection, name your database and collection, and insert your documents into your cluster.
    """
    logger.info("Set up our MongoDB connection, name your database and collection, and insert your documents into your cluster.")
    
    
    # connection_string = getpass.getpass(
        prompt="Enter connection string WITH USER + PASS here"
    )
    client = MongoClient(connection_string, appname="devrel.showcase.swig_menu")
    
    
    database = client["swig_menu"]
    collection = database["seasonal_drinks"]
    
    collection.insert_many(seasonal_drinks_pretty)
    
    """
    Create an Atlas Search index on your collection
    and create an aggregation pipeline. We are using the operator $search.
    
    Do NOT run this part in your notebook. This is done in the Atlas UI.
    
    This finds drinks that have "fall" in them
    """
    logger.info("Create an Atlas Search index on your collection")
    
    {"text": {"query": "fall", "path": "reason"}}
    
    """
    This finds drinks that are fall AND have apple as an ingredient
    """
    logger.info("This finds drinks that are fall AND have apple as an ingredient")
    
    {
        "compound": {
            "must": [
                {"text": {"query": "fall", "path": "reason"}},
                {"text": {"query": "apple", "path": "reason"}},
            ],
        }
    }
    
    """
    Now you can find drinks that are fall themed that are specific to any ingredients you want!
    """
    logger.info("Now you can find drinks that are fall themed that are specific to any ingredients you want!")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())