import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.graphql.base import GraphQLToolSpec
import os
import shutil


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
## GraphQL Agent Tool

This example walks through two examples of connecting an Agent to a GraphQL server, one unauthenticated endpoint and one authenticated. To start, we initialize the MLX package with our api key.
"""
logger.info("## GraphQL Agent Tool")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"


"""
## Unauthenticated server

Our first example is connecting to a server provided by Apollo as an introduction to GraphQL. It provides some data about SpaceX rockets and launches.

To get started, we setup the url we want to query and some basic headers, then we ask the agent to execute a query against the server.
"""
logger.info("## Unauthenticated server")


url = "https://spacex-production.up.railway.app/"
headers = {
    "content-type": "application/json",
}

graphql_spec = GraphQLToolSpec(url=url, headers=headers)
agent = FunctionAgent(
    tools=graphql_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_0ed3e03d():
    logger.debug(await agent.run("get the id, name and type of the Ships from the graphql endpoint"))
    return 
 = asyncio.run(run_async_code_0ed3e03d())
logger.success(format_json())

"""
The Agent was able to form the GraphQL based on our instructions, and additionally provided some extra parsing and formatting for the data. Nice!

## Authenticated Server

The next example shows setting up authentication headers to hit a private server, representing a Shopify store that has opened up GraphQL access based on an admin API token. To get started with an example similar to this, see the shopify.ipynb notebook. You will also find a more detailed example of using the Schema Definition Language file to fully unlock the GraphQL API.
"""
logger.info("## Authenticated Server")

url = "https://your-store.myshopify.com/admin/api/2023-07/graphql.json"
headers = {
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "X-Shopify-Access-Token": "your-admin-key",
}

graphql_spec = GraphQLToolSpec(url=url, headers=headers)
agent = FunctionAgent(
    tools=graphql_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

logger.debug(
    async def run_async_code_7fc48977():
        await agent.run("get the id and title of the first 3 products from the graphql server")
        return 
     = asyncio.run(run_async_code_7fc48977())
    logger.success(format_json())
)

logger.info("\n\n[DONE]", bright=True)