from IPython.display import Markdown, display
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import openai
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/JSONalyze_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# JSONalyze Query Engine

JSONalyze, or JSON Analyze Query Engine is designed to be wired typically after a calling(by agent, etc) of APIs, where we have the returned value as bulk instances of rows, and the next step is to perform statistical analysis on the data.

With JSONalyze, under the hood, in-memory SQLite table is created with the JSON List loaded, the query engine is able to perform SQL queries on the data, and return the Query Result as answer to the analytical questions.

This is a very simple example of how to use JSONalyze Query Engine.

First let's install llama-index.
"""
logger.info("# JSONalyze Query Engine")

# %pip install llama-index-llms-ollama

# %pip install llama-index

# %pip install sqlite-utils


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
# openai.api_key = os.environ["OPENAI_API_KEY"]


"""
Let's assume we have a list of JSON(already loaded as List of Dicts) as follows:
"""
logger.info("Let's assume we have a list of JSON(already loaded as List of Dicts) as follows:")

json_list = [
    {
        "name": "John Doe",
        "age": 25,
        "major": "Computer Science",
        "email": "john.doe@example.com",
        "address": "123 Main St",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "phone": "+1 123-456-7890",
        "occupation": "Software Engineer",
    },
    {
        "name": "Jane Smith",
        "age": 30,
        "major": "Business Administration",
        "email": "jane.smith@example.com",
        "address": "456 Elm St",
        "city": "San Francisco",
        "state": "CA",
        "country": "USA",
        "phone": "+1 234-567-8901",
        "occupation": "Marketing Manager",
    },
    {
        "name": "Michael Johnson",
        "age": 35,
        "major": "Finance",
        "email": "michael.johnson@example.com",
        "address": "789 Oak Ave",
        "city": "Chicago",
        "state": "IL",
        "country": "USA",
        "phone": "+1 345-678-9012",
        "occupation": "Financial Analyst",
    },
    {
        "name": "Emily Davis",
        "age": 28,
        "major": "Psychology",
        "email": "emily.davis@example.com",
        "address": "234 Pine St",
        "city": "Los Angeles",
        "state": "CA",
        "country": "USA",
        "phone": "+1 456-789-0123",
        "occupation": "Psychologist",
    },
    {
        "name": "Alex Johnson",
        "age": 27,
        "major": "Engineering",
        "email": "alex.johnson@example.com",
        "address": "567 Cedar Ln",
        "city": "Seattle",
        "state": "WA",
        "country": "USA",
        "phone": "+1 567-890-1234",
        "occupation": "Civil Engineer",
    },
    {
        "name": "Jessica Williams",
        "age": 32,
        "major": "Biology",
        "email": "jessica.williams@example.com",
        "address": "890 Walnut Ave",
        "city": "Boston",
        "state": "MA",
        "country": "USA",
        "phone": "+1 678-901-2345",
        "occupation": "Biologist",
    },
    {
        "name": "Matthew Brown",
        "age": 26,
        "major": "English Literature",
        "email": "matthew.brown@example.com",
        "address": "123 Peach St",
        "city": "Atlanta",
        "state": "GA",
        "country": "USA",
        "phone": "+1 789-012-3456",
        "occupation": "Writer",
    },
    {
        "name": "Olivia Wilson",
        "age": 29,
        "major": "Art",
        "email": "olivia.wilson@example.com",
        "address": "456 Plum Ave",
        "city": "Miami",
        "state": "FL",
        "country": "USA",
        "phone": "+1 890-123-4567",
        "occupation": "Artist",
    },
    {
        "name": "Daniel Thompson",
        "age": 31,
        "major": "Physics",
        "email": "daniel.thompson@example.com",
        "address": "789 Apple St",
        "city": "Denver",
        "state": "CO",
        "country": "USA",
        "phone": "+1 901-234-5678",
        "occupation": "Physicist",
    },
    {
        "name": "Sophia Clark",
        "age": 27,
        "major": "Sociology",
        "email": "sophia.clark@example.com",
        "address": "234 Orange Ln",
        "city": "Austin",
        "state": "TX",
        "country": "USA",
        "phone": "+1 012-345-6789",
        "occupation": "Social Worker",
    },
    {
        "name": "Christopher Lee",
        "age": 33,
        "major": "Chemistry",
        "email": "christopher.lee@example.com",
        "address": "567 Mango St",
        "city": "San Diego",
        "state": "CA",
        "country": "USA",
        "phone": "+1 123-456-7890",
        "occupation": "Chemist",
    },
    {
        "name": "Ava Green",
        "age": 28,
        "major": "History",
        "email": "ava.green@example.com",
        "address": "890 Cherry Ave",
        "city": "Philadelphia",
        "state": "PA",
        "country": "USA",
        "phone": "+1 234-567-8901",
        "occupation": "Historian",
    },
    {
        "name": "Ethan Anderson",
        "age": 30,
        "major": "Business",
        "email": "ethan.anderson@example.com",
        "address": "123 Lemon Ln",
        "city": "Houston",
        "state": "TX",
        "country": "USA",
        "phone": "+1 345-678-9012",
        "occupation": "Entrepreneur",
    },
    {
        "name": "Isabella Carter",
        "age": 28,
        "major": "Mathematics",
        "email": "isabella.carter@example.com",
        "address": "456 Grape St",
        "city": "Phoenix",
        "state": "AZ",
        "country": "USA",
        "phone": "+1 456-789-0123",
        "occupation": "Mathematician",
    },
    {
        "name": "Andrew Walker",
        "age": 32,
        "major": "Economics",
        "email": "andrew.walker@example.com",
        "address": "789 Berry Ave",
        "city": "Portland",
        "state": "OR",
        "country": "USA",
        "phone": "+1 567-890-1234",
        "occupation": "Economist",
    },
    {
        "name": "Mia Evans",
        "age": 29,
        "major": "Political Science",
        "email": "mia.evans@example.com",
        "address": "234 Lime St",
        "city": "Washington",
        "state": "DC",
        "country": "USA",
        "phone": "+1 678-901-2345",
        "occupation": "Political Analyst",
    },
]

"""
Then, we can create a JSONalyze Query Engine instance, with the JSON List as input.
"""
logger.info("Then, we can create a JSONalyze Query Engine instance, with the JSON List as input.")


llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

json_stats_query_engine = JSONalyzeQueryEngine(
    list_of_dict=json_list,
    llm=llm,
    verbose=True,
)

"""
To demonstrate the Query Engine, let's first create a list of stastical questions, and then we can use the Query Engine to answer the questions.
"""
logger.info("To demonstrate the Query Engine, let's first create a list of stastical questions, and then we can use the Query Engine to answer the questions.")

questions = [
    "What is the average age of the individuals in the dataset?",
    "What is the maximum age among the individuals?",
    "What is the minimum age among the individuals?",
    "How many individuals have a major in Psychology?",
    "What is the most common major among the individuals?",
    "What is the percentage of individuals residing in California (CA)?",
    "How many individuals have an occupation related to science or engineering?",
    "What is the average length of the email addresses in the dataset?",
    "How many individuals have a phone number starting with '+1 234'?",
    "What is the distribution of ages among the individuals?",
]

"""
Say we want to know the average of the age of the people in the list, we can use the following query:
"""
logger.info("Say we want to know the average of the age of the people in the list, we can use the following query:")

display(
    Markdown("> Question: {}".format(questions[0])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[0]))),
)

"""
We can also use the Query Engine to answer other questions:
"""
logger.info("We can also use the Query Engine to answer other questions:")

display(
    Markdown("> Question: {}".format(questions[4])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[4]))),
)

display(
    Markdown("> Question: {}".format(questions[7])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[7]))),
)

display(
    Markdown("> Question: {}".format(questions[5])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[5]))),
)

display(
    Markdown("> Question: {}".format(questions[9])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[9]))),
)

json_stats_aquery_engine = JSONalyzeQueryEngine(
    list_of_dict=json_list,
    llm=llm,
    verbose=True,
    use_async=True,
)

display(
    Markdown("> Question: {}".format(questions[7])),
    Markdown("Answer: {}".format(json_stats_query_engine.query(questions[7]))),
)

logger.info("\n\n[DONE]", bright=True)