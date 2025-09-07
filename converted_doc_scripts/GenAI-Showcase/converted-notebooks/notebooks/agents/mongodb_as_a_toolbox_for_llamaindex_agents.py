async def main():
    from jet.transformers.formatters import format_json
    from __main__ import decorated_tools_registry
    from datetime import datetime
    from functools import wraps
    from google.colab import userdata
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
    from jet.logger import CustomLogger
    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.memory import Memory
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.tools import FunctionTool
    from llama_index.embeddings.voyageai import VoyageEmbedding
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from typing import get_type_hints
    import inspect
    import os
    import pprint
    import pymongo
    import random
    import requests
    import shutil
    import string
    import time
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/chowgi/GenAI-Showcase/blob/main/mongodb_as_a_toolbox_for_llamaindex_agents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # MongoDB as a Toolbox for LlamaIndex Agents
    
    This notebook demonstrates how to leverage MongoDB Atlas as a "toolbox" for LlamaIndex agents. The application showcases the integration of MongoDB's capabilities, specifically its Vector Search feature, with LlamaIndex for building intelligent agents capable of performing various tasks by calling relevant tools stored and managed within MongoDB.
    
    **Key Features:**
    
    *   **MongoDB as a Tool Registry:** Instead of hardcoding tool definitions within the agent, this application stores tool metadata (name, description, parameters) directly in a MongoDB collection.
    *   **MongoDB Atlas Vector Search for Tool Discovery:** LlamaIndex uses the vector embeddings of tool descriptions stored in MongoDB to perform semantic searches based on user queries. This allows the agent to dynamically discover and select the most relevant tools for a given task.
    *   **LlamaIndex Agent with Function Calling:** The LlamaIndex agent is configured to use the retrieved tool definitions from MongoDB to enable function calling. This means the agent can understand the user's intent and execute the appropriate Python function (tool) stored in the application.
    *   **Data Storage in MongoDB:** Besides tool definitions, the application also uses separate MongoDB collections to store operational data like customer orders, return requests, and policy documents.
    *   **Integration with External Services:** The tools defined and managed in MongoDB can interact with external services (e.g., fetching real-time data, processing requests) or perform operations on the data stored within MongoDB itself (e.g., looking up order details, creating return requests).
    
    This approach provides a flexible and scalable way to manage and expand the agent's capabilities. New tools can be added to the MongoDB collection dynamically, and the agent can discover and utilize them without requiring code changes to the agent itself.
    
    # Environment Setup and Configuration
    
    This section covers the installation of necessary libraries, setting up API keys, and configuring the database connection to MongoDB Atlas.
    
    ### Install required libraries
    
    This cell installs the necessary Python libraries using `uv pip install`. These libraries include:
    - `pymongo`: A Python driver for MongoDB.
    - `llama-index-core`: The core LlamaIndex library.
    - `llama-index-llms-ollama`: LlamaIndex integration with Ollama LLMs.
    - `llama-index-embeddings-voyageai`: LlamaIndex integration with VoyageAI embeddings.
    - `llama-index-vector-stores-mongodb`: LlamaIndex integration with MongoDB Atlas Vector Search.
    - `llama-index-readers-file`: LlamaIndex file readers.
    """
    logger.info("# MongoDB as a Toolbox for LlamaIndex Agents")
    
    # !uv pip install pymongo llama-index-core llama-index-llms-ollama llama-index-embeddings-voyageai llama-index-vector-stores-mongodb llama-index-readers-file
    
    """
    ### Get and store API keys
    
    Get and store API keys
    This cell retrieves API keys for Ollama, MongoDB, and VoyageAI from Google Colab's user data secrets and sets them as environment variables.
    
    Please obtain your own API keys for Ollama, MongoDB Atlas, and VoyageAI.
    
    Ollama: You can get an API key from the Ollama website.
    MongoDB Atlas: Get your connection string from your MongoDB Atlas cluster.
    VoyageAI: Obtain an API key from the VoyageAI website.
    # Once you have your keys, add them to Google Colab's user data secrets by clicking on the "🔑" icon in the left sidebar. Name the secrets OPENAI_API_KEY, MONGODB_URI, and VOYAGE_API_KEY respectively.
    
    It also defines the GPT model to be used.
    """
    logger.info("### Get and store API keys")
    
    
    
    # OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    MONGO_URI = userdata.get("MONGODB_URI")
    os.environ["MONGO_URI"] = MONGO_URI
    
    VOYAGE_API_KEY = userdata.get("VOYAGE_API_KEY")
    os.environ["VOYAGE_API_KEY"] = VOYAGE_API_KEY
    
    GPT_MODEL = "gpt-4o"
    
    """
    ### Setup the database
    
    This cell establishes a connection to the MongoDB Atlas database using the provided URI. It then defines the database name and the names of the collections that will be used in this notebook for storing tools, orders, returns, and policies. Finally, it creates client objects for each of these collections.
    """
    logger.info("### Setup the database")
    
    
    mongo_client = pymongo.MongoClient(MONGO_URI, appname="showcase.tools.mongodb_toolbox")
    
    db_name = "retail_agent_demo"
    
    db = mongo_client[db_name]
    
    tools_collection_name = "tools"
    orders_collection_name = "orders"
    returns_collection_name = "returns"
    policies_collection_name = "policies"
    
    tools_collection = db[tools_collection_name]
    orders_collection = db[orders_collection_name]
    returns_collection = db[returns_collection_name]
    policies_collection = db[policies_collection_name]
    
    """
    # Loading Demo Data
    
    ## Download and store policy documents into MongoDB Atlas vector store
    
    This cell downloads policy documents and stores them in a MongoDB Atlas vector store. It initializes a vector store, checks if the collection is empty, downloads PDF documents, loads them, adds metadata, initializes embedding and node parsing, parses documents into nodes, creates a storage context, creates a vector index, and ingests the documents.
    """
    logger.info("# Loading Demo Data")
    
    
    
    # from llama_index.readers.file import PDFReader
    
    policy_vector_store = MongoDBAtlasVectorSearch(
        mongo_client,
        db_name=db_name,
        collection_name="policies",
        vector_index_name="vector_index",  # Assuming a vector index named 'vector_index' exists
    )
    
    policies_count = policy_vector_store.collection.count_documents({})
    if policies_count > 0:
        logger.debug(
            f"Policies collection is not empty. Skipping document import. Total documents: {policies_count}"
        )
    else:
        logger.debug("Policies collection is empty. Starting document import.")
    
        document_urls = [
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/privacy_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/return_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/shipping_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/terms_of_service.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/warranty_policy.pdf",
        ]
    
        temp_dir = "temp_policy_docs"
        os.makedirs(temp_dir, exist_ok=True)
    
        local_files = []
        for url in document_urls:
            file_name = os.path.join(temp_dir, url.split("/")[-1])
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                with open(file_name, "wb") as f:
                    f.write(response.content)
                local_files.append(file_name)
                logger.debug(f"Downloaded {url} to {file_name}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Error downloading {url}: {e}")
    
        documents = []
        for file_path in local_files:
            try:
    #             loader = PDFReader()
                docs = loader.load_data(file=file_path)
                documents.extend(docs)
                logger.debug(f"Loaded {file_path}")
            except Exception as e:
    #             logger.debug(f"Error loading {file_path} with PDFReader: {e}")
    
        for i, doc in enumerate(documents):
            doc.metadata.update(
                {
                    "document_type": "policy",
                    "document_index": i,
                    "file_name": os.path.basename(doc.metadata.get("file_path", "unknown")),
                }
            )
    
        logger.debug(f"Loaded {len(documents)} documents from directory")
    
        embed_model = VoyageEmbedding(model_name="voyage-3.5-lite", api_key=VOYAGE_API_KEY)
    
        node_parser = SentenceSplitter(chunk_size=2024, chunk_overlap=200)
    
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.debug(f"Created {len(nodes)} text chunks")
    
        storage_context = StorageContext.from_defaults(vector_store=policy_vector_store)
    
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
    
        logger.debug("Successfully ingested all PDF documents into MongoDB 'policies' collection")
    
        policies_count = policy_vector_store.collection.count_documents({})
        logger.debug(f"Total documents in 'policies' collection: {policies_count}")
    
    """
    ## Create and Store Dummy Order Data
    
    This cell generates a list of fake order data with details like order ID, date, status, total amount, shipping address, payment method, and items. It then checks if the `orders` collection in MongoDB is empty. If it is, the fake order data is inserted into the `orders` collection. This is done to populate the database with sample data for testing and demonstrating the order lookup functionality later in the notebook.
    """
    logger.info("## Create and Store Dummy Order Data")
    
    
    if orders_collection.count_documents({}) == 0:
        fake_orders = [
            {
                "order_id": 101,
                "order_date": datetime(2023, 10, 26, 10, 0, 0),
                "status": "Shipped",
                "total_amount": 150.75,
                "shipping_address": "123 Main St, Anytown, CA 91234",
                "payment_method": "Credit Card",
                "items": [
                    {"name": "Laptop", "price": 1200.00},
                    {"name": "Mouse", "price": 25.75},
                ],
            },
            {
                "order_id": 102,
                "order_date": datetime(2023, 10, 25, 14, 30, 0),
                "status": "Processing",
                "total_amount": 55.00,
                "shipping_address": "456 Oak Ave, Somewhere, NY 54321",
                "payment_method": "PayPal",
                "items": [
                    {"name": "Keyboard", "price": 75.00},
                ],
            },
            {
                "order_id": 103,
                "order_date": datetime(2023, 10, 25, 14, 30, 0),
                "status": "Processing",
                "total_amount": 35.00,
                "shipping_address": "789 Pine Rd, Elsewhere, TX 67890",
                "payment_method": "Debit Card",
                "items": [
                    {"name": "Monitor", "price": 250.00},
                ],
            },
        ]
    
        orders_collection.insert_many(fake_orders)
        logger.debug(f"Inserted {len(fake_orders)} fake orders.")
    else:
        logger.debug("Orders collection is not empty. Skipping insertion of fake orders.")
    
    """
    # Application Setup and Configuration
    
    ## Define MongoDB Tool Decorator
    
    This cell defines the `mongodb_toolbox` decorator. This decorator is used to register functions as tools that can be discovered and used by the LlamaIndex agent. It also handles generating embeddings for the tool descriptions and storing them in the MongoDB 'tools' collection for vector search.
    """
    logger.info("# Application Setup and Configuration")
    
    
    
    vector_store = MongoDBAtlasVectorSearch(
        mongo_client,
        db_name=db_name,
        collection_name="tools",
        vector_index_name="vector_index",
    )
    
    voyage_embed_model = VoyageEmbedding(
        model_name="voyage-3.5-lite",
    )
    
    decorated_tools_registry = {}
    
    
    def get_embedding(text):
        text = text.replace("\n", " ")
        return voyage_embed_model.get_text_embedding(text)
    
    
    def mongodb_toolbox(vector_store=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
    
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)
    
            tool_def = {
                "name": func.__name__,
                "description": docstring.strip(),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
    
            for param_name, param in signature.parameters.items():
                if (
                    param.kind == inspect.Parameter.VAR_POSITIONAL
                    or param.kind == inspect.Parameter.VAR_KEYWORD
                ):
                    continue
    
                param_type = type_hints.get(param_name, type(None))
                json_type = "string"  # Default to string
                if param_type in (int, float):
                    json_type = "number"
                elif param_type is bool:
                    json_type = "boolean"
    
                tool_def["parameters"]["properties"][param_name] = {
                    "type": json_type,
                    "description": f"Parameter {param_name}",
                }
    
                if param.default == inspect.Parameter.empty:
                    tool_def["parameters"]["required"].append(param_name)
    
            tool_def["parameters"]["additionalProperties"] = False
    
            document = Document(text=tool_def["description"], metadata=tool_def)
    
            if vector_store and tool_def["description"]:
                embedding = voyage_embed_model.get_text_embedding(tool_def["description"])
                document.embedding = embedding
    
            if vector_store:
                existing_doc = vector_store.collection.find_one(
                    {"metadata.name": tool_def["name"]}
                )
                if not existing_doc:
                    vector_store.add([document])
                else:
                    logger.debug(
                        f"Document for tool '{tool_def['name']}' already exists. Skipping insertion."
                    )
    
            decorated_tools_registry[func.__name__] = func
    
            return wrapper
    
        return decorator
    
    """
    ## Setup indexes
    
    This cell checks for and creates vector search indexes on the specified MongoDB collections if they don't already exist. These indexes are crucial for performing efficient vector searches on the data stored in these collections.
    """
    logger.info("## Setup indexes")
    
    
    required_indexs = [
        orders_collection_name,
        tools_collection_name,
        returns_collection_name,
        policies_collection_name,
    ]
    
    index_created = False
    
    for collection_name in required_indexs:
        logger.debug(f"Checking and creating index for collection: {collection_name}")
        current_vector_store = MongoDBAtlasVectorSearch(
            mongo_client,
            db_name=db_name,
            collection_name=collection_name,
            vector_index_name="vector_index",
        )
    
        try:
            search_indexes = list(current_vector_store.collection.list_search_indexes())
            index_exists = any(
                index.get("name") == "vector_index" for index in search_indexes
            )
        except Exception as e:
            logger.debug(f"Could not check search indexes for {collection_name}: {e}")
            index_exists = False
    
        if not index_exists:
            current_vector_store.create_vector_search_index(
                dimensions=1024, path="embedding", similarity="cosine"
            )
            logger.debug(f"Vector search index created successfully for {collection_name}.")
            index_created = True  # Set flag if an index was created
        else:
            logger.debug(
                f"Vector search index already exists for {collection_name}. Skipping creation."
            )
    
    if index_created:
        logger.debug("Pausing for 20 seconds to allow index builds...")
        time.sleep(20)
        logger.debug("Resuming after pause.")
    else:
        logger.debug("No new indexes were created. Skipping pause.")
    
    """
    ## Define Vector Search Function
    
    This cell defines the `vector_search_tools` function, which performs a vector search on a given LlamaIndex vector store based on a user query. It uses the specified vector store and embedding model to find the most relevant documents (in this case, tool definitions) and returns a list of their metadata.
    """
    logger.info("## Define Vector Search Function")
    
    def vector_search_tools(user_query, vector_store, top_k=3):
        """
        Perform a vector search using LlamaIndex vector store.
    
        Args:
            user_query (str): The user's query string.
            vector_store: The LlamaIndex vector store instance.
            top_k (int): Number of top results to return.
    
        Returns:
            list: A list of matching tool definitions.
        """
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=voyage_embed_model,
        )
    
        query_engine = index.as_query_engine(similarity_top_k=top_k)
    
        response = query_engine.query(user_query)
    
        tools_data = []
        for node in response.source_nodes:
            tool_metadata = node.node.metadata
            tools_data.append(tool_metadata)
    
        return tools_data
    
    """
    ## Define MongoDB Tools
    
    This cell defines several Python functions that will serve as tools for the LlamaIndex agent. Each function is decorated with the `@mongodb_toolbox` decorator, which registers the function and stores its definition and embedding in the 'tools' collection in MongoDB. These tools include functions for shouting, getting weather, getting stock price, getting current time, looking up orders, responding in Spanish, checking return policy, and creating a return request.
    """
    logger.info("## Define MongoDB Tools")
    
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def get_current_time(timezone: str = "UTC") -> str:
        """
        Get the current time for a specified timezone.
        Use this when a user asks about the current time in a specific timezone.
    
        :param timezone: The timezone to get the current time for. Defaults to 'UTC'.
        :return: A string with the current time in the specified timezone.
        """
        current_time = datetime.utcnow().strftime("%H:%M:%S")
        return f"The current time in {timezone} is {current_time}."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def lookup_order_number(order_id: int) -> str:
        """
        Lookup the details of a specific order number using its order ID.
        Use this when a user asks for information about a particular order.
    
        :param order_id: The unique identifier of the order to look up.
        :return: A string containing the order details or a message if the order is not found.
        """
        orders_collection = db["orders"]
    
        order = orders_collection.find_one(
            {"order_id": order_id}
        )  # Note: Sample data uses 'order_id'
    
        if order:
            order_details = f"Order ID: {order.get('order_id')}\n"
            order_details += (
                f"Order Date: {order.get('order_date').strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            order_details += f"Status: {order.get('status')}\n"
            order_details += f"Total Amount: ${order.get('total_amount')}\n"
            order_details += f"Shipping Address: {order.get('shipping_address')}\n"
            order_details += f"Payment Method: {order.get('payment_method')}\n"
            order_details += "Items:\n"
            for item in order.get("items", []):
                order_details += f"- {item.get('name')}: ${item.get('price')}\n"
    
            return order_details
        else:
            return f"Order with ID {order_id} not found."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def return_policy(return_request_description: str) -> str:
        """
        Performs search on the policies collection to determine if a user's
        return request aligns with the company's return policy, warranty policy,
        or terms of service. Use this tool when a user asks about returning an item
        and you need to check the relevant company policies.
    
        Args:
            return_request_description (str): A detailed description of the user's
                                              return request, including reasons
                                              for return, item condition, and any
                                              relevant order information.
    
        Returns:
            str: A string containing relevant policy information found through
                 vector search that can help determine if the return request
                 meets the company's policy.
        """
        policy_index = VectorStoreIndex.from_vector_store(
            policy_vector_store,
            embed_model=voyage_embed_model,
        )
    
        policy_query_engine = policy_index.as_query_engine(
            similarity_top_k=3
        )  # Adjust top_k as needed
    
        response = policy_query_engine.query(return_request_description)
    
        return str(response)
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def create_return_request(order_id: int, reason: str) -> str:
        """
        Creates a return request entry for a given order ID with the specified reason.
        Use this when a user wants to initiate a return for an item from a specific order.
    
        :param order_id: The unique identifier of the order for which the return is requested.
        :param reason: The reason for the return.
        :return: A string confirming the return creation or indicating if the order was not found.
        """
        orders_collection = db["orders"]
        returns_collection = db["returns"]
    
        order = orders_collection.find_one({"order_id": order_id})
    
        if order:
            return_data = {
                "return_id": returns_collection.count_documents({})
                + 1,  # Simple auto-incrementing ID
                "order_id": order_id,
                "return_date": datetime.utcnow(),
                "reason": reason,
                "status": "Pending",  # Initial status
                "items": order.get("items", []),  # Include items from the original order
            }
    
            returns_collection.insert_one(return_data)
    
            return f"Return request created successfully for order {order_id} with reason: {reason}."
        else:
            return f"Order with ID {order_id} not found. Could not create return."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def greet_user(name: str) -> str:
        """
        Greets the user by name.
        Use this when a user provides their name and you want to greet them.
    
        :param name: The name of the user.
        :return: A greeting message.
        """
        return f"Hello, {name}! Nice to meet you."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def calculate_square_root(number: float) -> str:
        """
        Calculates the square root of a given number.
        Use this when a user asks for the square root of a number.
    
        :param number: The number for which to calculate the square root.
        :return: A string with the square root result.
        """
        if number < 0:
            return "Cannot calculate the square root of a negative number."
        return f"The square root of {number} is {number**0.5}."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def repeat_phrase(phrase: str, times: int = 1) -> str:
        """
        Repeats a given phrase a specified number of times.
        Use this when a user asks you to repeat something.
    
        :param phrase: The phrase to repeat.
        :param times: The number of times to repeat the phrase. Defaults to 1.
        :return: A string with the repeated phrase.
        """
        if times <= 0:
            return "Please specify a positive number of times to repeat."
        return (phrase + " ") * times
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def roll_dice(number_of_dice: int = 1, sides: int = 6) -> str:
        """
        Rolls a specified number of dice with a given number of sides and returns the results.
        Use this when a user asks to roll dice.
    
        :param number_of_dice: The number of dice to roll. Defaults to 1.
        :param sides: The number of sides on each die. Defaults to 6.
        :return: A string showing the result of each roll and the total.
        """
        if number_of_dice <= 0 or sides <= 0:
            return "Please specify a positive number of dice and sides."
        rolls = [random.randint(1, sides) for _ in range(number_of_dice)]
        total = sum(rolls)
        return f"You rolled {number_of_dice} dice with {sides} sides each. Results: {rolls}. Total: {total}."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def flip_coin(number_of_flips: int = 1) -> str:
        """
        Flips a coin a specified number of times and returns the results.
        Use this when a user asks to flip a coin.
    
        :param number_of_flips: The number of times to flip the coin. Defaults to 1.
        :return: A string showing the result of each flip.
        """
        if number_of_flips <= 0:
            return "Please specify a positive number of flips."
        results = [random.choice(["Heads", "Tails"]) for _ in range(number_of_flips)]
        return f"You flipped the coin {number_of_flips} times. Results: {results}."
    
    
    @mongodb_toolbox(vector_store=vector_store)
    def generate_random_password(length: int = 12) -> str:
        """
        Generates a random password of a specified length.
        Use this when a user asks for a random password.
    
        :param length: The desired length of the password. Defaults to 12.
        :return: A randomly generated password string.
        """
        if length <= 0:
            return "Please specify a positive password length."
    
        characters = string.ascii_letters + string.digits + string.punctuation
        password = "".join(random.choice(characters) for i in range(length))
        return f"Here is a random password: {password}"
    
    """
    ## Populate Tools Function
    
    This cell defines the `populate_tools` function. This function takes the results from a vector search (which are tool definitions) and converts them into a list of LlamaIndex `FunctionTool` objects. It looks up the actual function object in the `decorated_tools_registry` based on the tool name found in the search results and creates a `FunctionTool` with the corresponding function and its description.
    """
    logger.info("## Populate Tools Function")
    
    
    
    
    def populate_tools(search_results):
        """
        Populate the tools array based on the results from the vector search,
        returning LlamaIndex FunctionTool objects.
    
        Args:
        search_results (list): The list of documents returned from the vector search.
    
        Returns:
        list: A list of LlamaIndex FunctionTool objects.
        """
        tools = []
    
        for result in search_results:
            tool_name = result["name"]
            function_obj = decorated_tools_registry.get(tool_name)
    
            if function_obj:
                description = result.get("description", function_obj.__doc__ or "")
                tools.append(
                    FunctionTool.from_defaults(fn=function_obj, description=description)
                )
        return tools
    
    """
    # Running the Agent
    
    ## Test Tool Retrieval
    
    This cell demonstrates how to use the `vector_search_tools` function to find relevant tools based on a user query and then uses the `populate_tools` function to convert the search results into LlamaIndex `FunctionTool` objects. Finally, it prints the names of the retrieved tools to verify the process.
    """
    logger.info("# Running the Agent")
    
    
    user_query = (
        "I want to return a damaged laptop from order 101. What is the return policy??"
    )
    
    tools_related_to_user_query = vector_search_tools(user_query, vector_store)
    
    tools = populate_tools(tools_related_to_user_query)
    
    
    tool_names = [tool.metadata.name for tool in tools]
    logger.debug(
        "Selected tools from the toolbox based on the similarity to the users intention -"
    )
    pprint.plogger.debug(tool_names)
    
    
    llm = Ollama(model=GPT_MODEL)
    
    tools = populate_tools(tools_related_to_user_query)
    
    memory = Memory.from_defaults(session_id="my_session", token_limit=40000)
    
    agent = FunctionAgent(llm=llm, tools=tools)
    
    response = await agent.run("How much did I pay for order 101", memory=memory)
    logger.success(format_json(response))
    logger.debug(response)
    
    response = await agent.run(
            "I want to return a damaged laptop from order 101. What is the return policy??",
            memory=memory,
        )
    logger.success(format_json(response))
    logger.debug(response)
    
    response = await agent.run("Yes please", memory=memory)
    logger.success(format_json(response))
    logger.debug(response)
    
    """
    ### Get and store API keys
    
    This cell retrieves API keys for Ollama, MongoDB, and VoyageAI from Google Colab's user data secrets and sets them as environment variables.
    
    **Please obtain your own API keys for Ollama, MongoDB Atlas, and VoyageAI.**
    
    *   **Ollama:** You can get an API key from the [Ollama website](https://platform.ollama.com/).
    *   **MongoDB Atlas:** Get your connection string from your MongoDB Atlas cluster.
    *   **VoyageAI:** Obtain an API key from the [VoyageAI website](https://voyageai.com/).
    
    # Once you have your keys, add them to Google Colab's user data secrets by clicking on the "🔑" icon in the left sidebar. Name the secrets `OPENAI_API_KEY`, `MONGODB_URI`, and `VOYAGE_API_KEY` respectively.
    
    It also defines the GPT model to be used.
    """
    logger.info("### Get and store API keys")
    
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