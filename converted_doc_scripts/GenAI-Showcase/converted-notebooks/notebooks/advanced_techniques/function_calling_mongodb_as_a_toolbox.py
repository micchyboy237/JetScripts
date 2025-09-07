from datetime import datetime
from functools import wraps
from jet.logger import CustomLogger
from typing import get_type_hints
import inspect
import json
import ollama
import os
import pprint
import pymongo
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# MongoDB As A Toolbox For Agentic Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/function_calling_mongodb_as_a_toolbox.ipynb)
"""
logger.info("# MongoDB As A Toolbox For Agentic Systems")

# !pip install --quiet ollama pymongo

# import getpass

# OPENAI_API_KEY = getpass.getpass("Ollama API Key: ")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# MONGO_URI = getpass.getpass("Enter MongoDB URI: ")
os.environ["MONGO_URI"] = MONGO_URI

GPT_MODEL = "gpt-4o"


client = ollama.Ollama()

"""
## Define MongoDB Tool Decorator
"""
logger.info("## Define MongoDB Tool Decorator")


mongo_client = pymongo.MongoClient(MONGO_URI, appname="showcase.tools.mongodb_toolbox")

db = mongo_client["function_calling_db"]

tools_collection = db["tools"]



def get_embedding(text, model="mxbai-embed-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def mongodb_toolbox(collection=tools_collection):
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

        vector = get_embedding(tool_def["description"])
        tool_doc = {**tool_def, "embedding": vector}
        collection.update_one({"name": func.__name__}, {"$set": tool_doc}, upsert=True)

        return wrapper

    return decorator

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 2,  # Return top 5 matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    pipeline = [vector_search_stage, unset_stage]

    results = collection.aggregate(pipeline)
    return list(results)



@mongodb_toolbox()
def shout(statement: str) -> str:
    """
    Convert a statement to uppercase letters to emulate shouting. Use this when a user wants to emphasize something strongly or when they explicitly ask to 'shout' something..

    """
    return statement.upper()


@mongodb_toolbox()
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a specified location.
    Use this when a user asks about the weather in a specific place.

    :param location: The name of the city or location to get weather for.
    :param unit: The temperature unit, either 'celsius' or 'fahrenheit'. Defaults to 'celsius'.
    :return: A string describing the current weather.
    """
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temperature = random.randint(-10, 35)

    if unit.lower() == "fahrenheit":
        temperature = (temperature * 9 / 5) + 32

    condition = random.choice(conditions)
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if unit.lower() == 'celsius' else 'F'}."


@mongodb_toolbox()
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a given stock symbol.
    Use this when a user asks about the current price of a specific stock.

    :param symbol: The stock symbol to look up (e.g., 'AAPL' for Apple Inc.).
    :return: A string with the current stock price.
    """
    price = round(random.uniform(10, 1000), 2)
    return f"The current stock price of {symbol} is ${price}."


@mongodb_toolbox()
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time for a specified timezone.
    Use this when a user asks about the current time in a specific timezone.

    :param timezone: The timezone to get the current time for. Defaults to 'UTC'.
    :return: A string with the current time in the specified timezone.
    """
    current_time = datetime.utcnow().strftime("%H:%M:%S")
    return f"The current time in {timezone} is {current_time}."

def populate_tools(search_results):
    """
    Populate the tools array based on the results from the vector search.

    Args:
    search_results (list): The list of documents returned from the vector search.

    Returns:
    list: A list of tool definitions in the format required by the Ollama API.
    """
    tools = []
    for result in search_results:
        tool = {
            "type": "function",
            "function": {
                "name": result["name"],
                "description": result["description"],
                "parameters": result["parameters"],
            },
        }
        tools.append(tool)
    return tools

user_query = "Hi, can you shout the statement: We are there"

tools_related_to_user_query = vector_search(user_query, tools_collection)

tools = populate_tools(tools_related_to_user_query)


pprint.plogger.debug(tools)

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    },
    {"role": "user", "content": user_query},
]

response = ollama.chat.completions.create(
    model=GPT_MODEL,
    messages=messages,
    tools=tools,
)

response_message = response.choices[0].message
messages.append(response_message)

logger.debug(response_message)

tool_calls = response_message.tool_calls
if tool_calls:
    tool_call = tool_calls[0]
    tool_call_id = tool_call.id
    tool_function_name = tool_call.function.name

    logger.debug(f"Debug - Tool call received: {tool_function_name}")
    logger.debug(f"Debug - Arguments: {tool_call.function.arguments}")

    try:
        tool_arguments = json.loads(tool_call.function.arguments)
        tool_query_string = tool_arguments.get("statement", "")
    except json.JSONDecodeError:
        logger.debug(
            f"Error: Unable to parse function arguments: {tool_call.function.arguments}"
        )
        tool_query_string = ""

    if tool_function_name == "shout":
        results = shout(tool_query_string)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_function_name,
                "content": results,
            }
        )

        model_response_with_function_call = client.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        logger.debug(model_response_with_function_call.choices[0].message.content)
    else:
        logger.debug(f"Error: function {tool_function_name} does not exist")
else:
    logger.debug(response_message.content)

logger.info("\n\n[DONE]", bright=True)