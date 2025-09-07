async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from playwright.async_api import async_playwright
    from pymongo.operations import SearchIndexModel
    import os
    import pandas as pd
    import pymongo
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/TraderJoesFallAIPartyPlanner_PlaywrightLlamaIndexVectorSearch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    [![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/mongodb/trader-joes-llamaindex-vector-search/)
    
    ## Overview
    In this tutorial we are going to scrape the Trader Joe's website for all their Fall Faves using Playwright, and then create an AI party planner using the LlamaIndex x MongoDB Vector Search integration and a chat engine. This will help us figure out easily which Trader Joe's fall faves would be perfect for our fall festivities.
    
    What's Covered
    
    *   Building a Trader Joe’s AI party planner using Playwright, LlamaIndex, and  MongoDB Atlas Vector Search
    *   Scraping Trader Joe’s fall items with Playwright and formatting them for chatbot use
    *   Setting up and embedding product data in MongoDB Atlas Vector Store for semantic search
    *   Creating a Retrieval-Augmented Generation (RAG) chatbot to answer party planning questions
    *   Adding interactive Chat Engine functionality for back-and-forth Q&A about fall party items
    
    ## Part 1: Scrape the Trader Joes website for their fall items
    
    First, let’s go ahead and install Playwright:
    """
    logger.info("## Overview")
    
    # !pip install playwright
    # !playwright install
    
    """
    Once that’s done installing, we can go ahead and import our necessary packages:
    """
    logger.info("Once that’s done installing, we can go ahead and import our necessary packages:")
    
    
    """
    Please keep in mind that we are using `async` because we are running everything inside of a Google Colab notebook.
    
    Now, let’s start building our `traderJoesScraper`:
    
    We started off with manually putting in all the links we want to scrape the information off of, please keep in mind that if you’re hoping to turn this into a scalable application it’s recommended to use pagination for this part, but for the sake of simplicity, we can input them manually.
    
    Then we just looped through each of the URL’s listed, waited for our main selector to show up that had all the elements we were hoping to scrape, and then extracted our “name” and “price”.
    
    Once we ran that, we got a list of all our products from the Fall Faves tag!
    """
    logger.info("Please keep in mind that we are using `async` because we are running everything inside of a Google Colab notebook.")
    
    async def traderJoesScraper():
        async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                page = await browser.new_page()
            
                pages = [
                    {
                        "url": "https://www.traderjoes.com/home/products/category/food-8?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%7D",
                        "category": "Food",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/food-8?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%2C%22page%22%3A2%7D",
                        "category": "Food",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/food-8?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%2C%22page%22%3A3%7D",
                        "category": "Food",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/food-8?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%2C%22page%22%3A4%7D",
                        "category": "Food",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/food-8?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%2C%22page%22%3A5%7D",
                        "category": "Food",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/beverages-182?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%7D",
                        "category": "Beverage",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/flowers-plants-203?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%7D",
                        "category": "Flowers&Plants",
                    },
                    {
                        "url": "https://www.traderjoes.com/home/products/category/everything-else-215?filters=%7B%22tags%22%3A%5B%22Fall+Faves%22%5D%7D",
                        "category": "EverythingElse",
                    },
                ]
            
                items = []
            
                for info in pages:
                    await page.goto(info["url"])
            
                    await page.wait_for_selector(
                        "li.ProductList_productList__item__1EIvq",
                        state="attached",
                        timeout=60000,
                    )
            
                    products = await page.query_selector_all(
                        "li.ProductList_productList__item__1EIvq"
                    )
            
                    for product in products:
                        result = {}
            
                        name = await product.query_selector(
                            "h2.ProductCard_card__title__text__uiWLe a"
                        )
                        price = await product.query_selector(
                            "span.ProductPrice_productPrice__price__3-50j"
                        )
            
                        if name and price:
                            result["name"] = await name.inner_text()
            
                            price_text = await price.inner_text()
                            convert_price = float(price_text.replace("$", "").strip())
                            result["price"] = convert_price
            
                            result["category"] = info["category"]
                            items.append(result)
            
                for item in items:
                    logger.debug(
                        f"Name: {item['name']}, Price: {item['price']}, Category: {item['category']}"
                    )
            
                await browser.close()
                return items
            
            
        logger.success(format_json(result))
    scraped_products = await traderJoesScraper()
    logger.success(format_json(scraped_products))
    logger.debug(scraped_products)
    
    """
    To keep track of the items, we can go ahead and quickly count them:
    """
    logger.info("To keep track of the items, we can go ahead and quickly count them:")
    
    scraped_products_count = len(scraped_products)
    logger.debug(scraped_products_count)
    
    """
    As of the date this was scrapped we had 89 products.
    
    Now, let’s go ahead and save our products into a `.txt` file so we can use it later in our tutorial when we are using our LlamaIndex and Atlas Vector Search integration. Go ahead and name the file whatever you like, for sake of tracking I’m naming mine: `tj_fall_faves_oct30.txt`.
    """
    logger.info("As of the date this was scrapped we had 89 products.")
    
    with open("tj_fall_faves_oct30.txt", "w") as f:
        for item in scraped_products:
            f.write(
                f"Name: {item['name']}, Price: ${item['price']}, Category: {item['category']}\n"
            )
    
    
    df = pd.DataFrame(scraped_products)
    
    csv_path = "tj_fall_faves_oct30.csv"
    df.to_csv(csv_path, index=False)
    
    """
    Since we are using a notebook, please make sure that you download the file locally, since once our runtime is disconnected the `.txt` file will be lost.
    
    Now that we have all our Trader Joe’s fall products let’s go ahead and build out our AI Party Planner!
    
    ## Part 2: LlamaIndex and Atlas Vector Search Integration
    
    This is the quickstart we are using in order to be successful with this part of the tutorial: https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/llamaindex/#:~:text=You%20can%20integrate%20Atlas%20Vector,RAG). We will be going over how to use Atlas Vector Search with LlamaIndex to build a RAG application with chat capabilities!
    
    This section will cover in detail how to set up the environment, store our custom data that we previously scraped on Atlas, create an Atlas Vector Search index on top of our data, and to finish up we will implement RAG and will use Atlas Vector Search to answer questions from our unique data store.
    
    
    Let’s first use `pip` to install all our necessary libraries. We will need to include `llama-index`, `llama-index-vector-stores-mongodb`, and `llama-index-embeddings-huggingface`.
    """
    logger.info("## Part 2: LlamaIndex and Atlas Vector Search Integration")
    
    pip install --quiet --upgrade llama-index llama-index-vector-stores-mongodb llama-index-embeddings-huggingface pymongo
    
    """
    Now go ahead and import in your necessary import statements:
    """
    logger.info("Now go ahead and import in your necessary import statements:")
    
    # import getpass
    
    
    """
    Input your Ollama API Key and your MongoDB Atlas cluster connection string when prompted:
    """
    logger.info("Input your Ollama API Key and your MongoDB Atlas cluster connection string when prompted:")
    
    # os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
    # ATLAS_CONNECTION_STRING = getpass.getpass("MongoDB Atlas SRV Connection String:")
    
    """
    Once your keys are in, let’s go ahead and assign our specific models for `llama_index` so it knows how to properly embed our file. This is just to keep everything consistent!
    """
    logger.info("Once your keys are in, let’s go ahead and assign our specific models for `llama_index` so it knows how to properly embed our file. This is just to keep everything consistent!")
    
    Settings.llm = Ollama()
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    
    """
    Now we can go ahead and read in our `.txt` file with our scraped products. We are doing this using the `SimpleDirectoryReader` from `llama_index`. Text files aren’t the only files that can be nicely loaded into LlamaIndex. There are a ton of other supported methods and I recommend checking out some of their [supported file types](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?gad_source=1&gclid=Cj0KCQjwsoe5BhDiARIsAOXVoUsbgqjQcjmkV_KFLzS0TwUcONhaXfTaVT-C71A8Py_dHPHHSs-hmMsaAsbaEALw_wcB).
    
    So here we are just reading the contents of our file and then returning it as a list of documents; the format LlamaIndex requires.
    """
    logger.info("Now we can go ahead and read in our `.txt` file with our scraped products. We are doing this using the `SimpleDirectoryReader` from `llama_index`. Text files aren’t the only files that can be nicely loaded into LlamaIndex. There are a ton of other supported methods and I recommend checking out some of their [supported file types](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?gad_source=1&gclid=Cj0KCQjwsoe5BhDiARIsAOXVoUsbgqjQcjmkV_KFLzS0TwUcONhaXfTaVT-C71A8Py_dHPHHSs-hmMsaAsbaEALw_wcB).")
    
    sample_data = SimpleDirectoryReader(
        input_files=["/content/tj_fall_faves_oct30.txt"]
    ).load_data()
    sample_data[0]
    
    """
    Now that our file has been read in, let’s connect to our MongoDB Atlas cluster and set up a vector store! Feel free to name the database and collection anything you like. We are initializing a vector store using `MongoAtlasVectorSearch` from `llama_index` which will allow us to work with our embedded documents directly in our cluster.
    """
    logger.info("Now that our file has been read in, let’s connect to our MongoDB Atlas cluster and set up a vector store! Feel free to name the database and collection anything you like. We are initializing a vector store using `MongoAtlasVectorSearch` from `llama_index` which will allow us to work with our embedded documents directly in our cluster.")
    
    mongo_client = pymongo.MongoClient(
        ATLAS_CONNECTION_STRING, appname="devrel.showcase.tj_fall_faves"
    )
    
    atlas_vector_store = MongoDBAtlasVectorSearch(
        mongo_client,
        db_name="tj_products",
        collection_name="fall_faves",
        vector_index_name="vector_index",
    )
    vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_store)
    
    """
    Since our vector store has been defined (by our `vector_store_context`) let’s go ahead and create a vector index in MongoDB for our documents in `sample_data`.
    """
    logger.info("Since our vector store has been defined (by our `vector_store_context`) let’s go ahead and create a vector index in MongoDB for our documents in `sample_data`.")
    
    vector_store_index = VectorStoreIndex.from_documents(
        sample_data, storage_context=vector_store_context, show_progress=True
    )
    
    """
    Once this cell has run you can go ahead and view your data with the embeddings inside of your Atlas cluster.
    
    In order to allow for vector search queries on our created vector store, we need to create an Atlas Vector Search index on our tj_products.fall_faves collection. We can do this either through the [Atlas UI](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/) or directly from our notebook:
    """
    logger.info("Once this cell has run you can go ahead and view your data with the embeddings inside of your Atlas cluster.")
    
    collection = mongo_client["tj_products"]["fall_faves"]
    
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "metadata.page_label"},
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    
    collection.create_search_index(model=search_index_model)
    
    """
    You’ll be able to see this index once it’s up and running under your “Atlas Search” tab in your Atlas UI. Once it’s done, we can start querying our data and we can do some basic RAG.
    
    ## Part 3: Basic RAG
    
    With our Atlas Vector Search index up and running we are ready to have some fun and bring our AI Party Planner to life! We are going to continue with this dream team where we will use Atlas Vector Search to get our documents and LlamaIndex’s query engine to actually answer our questions based on our documents.
    
    To do this, we will need to have Atlas Vector Search become a [vector index retriever](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/vector/) and we will need to initialize a `RetrieverQueryEngine` to handle queries by passing each question through our vector retrieval system. This combination will allow us to ask any questions we want in natural language, and it will match us with the most accurate documents.
    """
    logger.info("## Part 3: Basic RAG")
    
    vector_store_retriever = VectorIndexRetriever(
        index=vector_store_index, similarity_top_k=5
    )
    
    query_engine = RetrieverQueryEngine(retriever=vector_store_retriever)
    
    response = query_engine.query(
        "Which plant items are available right now? Please provide prices"
    )
    
    logger.debug(response)
    
    """
    But what if we want to keep asking questions and get responses with memory? Let’s quickly build a [Chat Engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/).
    
    ## Part 4: Chat engine
    
    Instead of having to ask one question at a time about our Trader Joe’s products for our party, we can go ahead and incorporate a back-and-forth conversation to get the most out of our AI Party Planner.
    
    We first need to initialize the chat engine from our `vector_store_index` and enable a streaming response. [Condense question mode](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_question/) is also used to ensure that the engine shortens their questions or rephrases them to make the most sense when used in a back and forth conversation. Streaming is enabled as well so we can see the response:
    """
    logger.info("## Part 4: Chat engine")
    
    chat_engine = vector_store_index.as_chat_engine(
        chat_mode="condense_question", streaming=True
    )
    
    while True:
        question = input("Ask away! Type 'exit' to quit >>> ")
    
        if question == "exit":
            logger.debug("Exiting chat. Have a happy fall!")
            break
    
        logger.debug("\n")
    
        response_stream = chat_engine.stream_chat(question)
    
        response_stream.print_response_stream()
        logger.debug("\n")
    
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