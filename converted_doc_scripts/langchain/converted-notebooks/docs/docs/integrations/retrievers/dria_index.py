from jet.logger import logger
from langchain_community.retrievers import DriaRetriever
import os
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
# Dria

>[Dria](https://dria.co/) is a hub of public RAG models for developers to both contribute and utilize a shared embedding lake. This notebook demonstrates how to use the `Dria API` for data retrieval tasks.

# Installation

Ensure you have the `dria` package installed. You can install it using pip:
"""
logger.info("# Dria")

# %pip install --upgrade --quiet dria

"""
# Configure API Key

Set up your Dria API key for access.
"""
logger.info("# Configure API Key")


os.environ["DRIA_API_KEY"] = "DRIA_API_KEY"

"""
# Initialize Dria Retriever

Create an instance of `DriaRetriever`.
"""
logger.info("# Initialize Dria Retriever")


api_key = os.getenv("DRIA_API_KEY")
retriever = DriaRetriever(api_key=api_key)

"""
# **Create Knowledge Base**

Create a knowledge on [Dria's Knowledge Hub](https://dria.co/knowledge)
"""
logger.info("# **Create Knowledge Base**")

contract_id = retriever.create_knowledge_base(
    name="France's AI Development",
    embedding=DriaRetriever.models.jina_embeddings_v2_base_en.value,
    category="Artificial Intelligence",
    description="Explore the growth and contributions of France in the field of Artificial Intelligence.",
)

"""
# Add Data

Load data into your Dria knowledge base.
"""
logger.info("# Add Data")

texts = [
    "The first text to add to Dria.",
    "Another piece of information to store.",
    "More data to include in the Dria knowledge base.",
]

ids = retriever.add_texts(texts)
logger.debug("Data added with IDs:", ids)

"""
# Retrieve Data

Use the retriever to find relevant documents given a query.
"""
logger.info("# Retrieve Data")

query = "Find information about Dria."
result = retriever.invoke(query)
for doc in result:
    logger.debug(doc)

logger.info("\n\n[DONE]", bright=True)