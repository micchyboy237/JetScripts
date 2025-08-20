import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.database.base import DatabaseToolSpec
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
## MLX 

For this notebook we will use the MLX ChatGPT models. We import the MLX agent and set the api_key, you will have to provide your own API key.
"""
logger.info("## MLX")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"


"""
## Database tool

This tool connects to a database (using SQLAlchemy under the hood) and allows an Agent to query the database and get information about the tables.

We import the ToolSpec and initialize it so that it can connect to our database
"""
logger.info("## Database tool")


db_spec = DatabaseToolSpec(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="x",  # Database Password
    dbname="your_db",  # Database Name
)

"""
After initializing the Tool Spec we have an instance of the ToolSpec. A ToolSpec can have many tools that it implements and makes available to agents. We can see the Tools by converting to the spec to a list of FunctionTools, using `to_tool_list`
"""
logger.info("After initializing the Tool Spec we have an instance of the ToolSpec. A ToolSpec can have many tools that it implements and makes available to agents. We can see the Tools by converting to the spec to a list of FunctionTools, using `to_tool_list`")

tools = db_spec.to_tool_list()
for tool in tools:
    logger.debug(tool.metadata.name)
    logger.debug(tool.metadata.description)
    logger.debug(tool.metadata.fn_schema)

"""
We can see that the database tool spec provides 3 functions for the MLX agent. One to execute a SQL query, one to describe a list of tables in the database, and one to list all of the tables available in the database. 

We can pass the tool list to our MLX agent and test it out:
"""
logger.info("We can see that the database tool spec provides 3 functions for the MLX agent. One to execute a SQL query, one to describe a list of tables in the database, and one to list all of the tables available in the database.")

agent = FunctionAgent(
    tools=db_tools.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

ctx = Context(agent)

"""
At this point our Agent is fully ready to start making queries to our database:
"""
logger.info("At this point our Agent is fully ready to start making queries to our database:")

async def run_async_code_eb3249dd():
    logger.debug(await agent.run("What tables does this database contain", ctx=ctx))
    return 
 = asyncio.run(run_async_code_eb3249dd())
logger.success(format_json())

async def run_async_code_e1c1faa7():
    logger.debug(await agent.run("Can you describe the messages table", ctx=ctx))
    return 
 = asyncio.run(run_async_code_e1c1faa7())
logger.success(format_json())

async def run_async_code_2d773bf0():
    logger.debug(await agent.run("Fetch the most recent message and display the body", ctx=ctx))
    return 
 = asyncio.run(run_async_code_2d773bf0())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)