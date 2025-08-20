import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.metaphor.base import MetaphorToolSpec
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
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
# Building a Metaphor Data Agent

This tutorial walks through using the LLM tools provided by the [Metaphor API](https://platform.metaphor.systems/) to allow LLMs to easily search and retrieve HTML content from the Internet.

To get started, you will need an [MLX api key](https://platform.openai.com/account/api-keys) and a [Metaphor API key](https://dashboard.metaphor.systems/overview)

We will import the relevant agents and tools and pass them our keys here:
"""
logger.info("# Building a Metaphor Data Agent")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"



metaphor_tool = MetaphorToolSpec(
    api_key="your-key",
)

metaphor_tool_list = metaphor_tool.to_tool_list()
for tool in metaphor_tool_list:
    logger.debug(tool.metadata.name)

"""
## Testing the Metaphor tools

We've imported our MLX agent, set up the api key, and initialized our tool, checking the methods that it has available. Let's test out the tool before setting up our Agent.

All of the Metaphor search tools make use of the `AutoPrompt` option where Metaphor will pass the query through an LLM to refine and improve it.
"""
logger.info("## Testing the Metaphor tools")

metaphor_tool.search("machine learning transformers", num_results=3)

metaphor_tool.retrieve_documents(["iEYMai5rS9k0hN5_BH0VZg"])

metaphor_tool.find_similar(
    "https://www.mihaileric.com/posts/transformers-attention-in-disguise/"
)

metaphor_tool.search_and_retrieve_documents(
    "This is the best explanation for machine learning transformers:", num_results=1
)

"""
We can see we have different tools to search for results, retrieve the results, find similar results to a web page, and finally a tool that combines search and document retrieval into a single tool. We will test them out in LLM Agents below:

### Using the Search and Retrieve documents tools in an Agent

We can create an agent with access to the above tools and start testing it out:
"""
logger.info("### Using the Search and Retrieve documents tools in an Agent")

agent = FunctionAgent(
    tools=metaphor_tool_list,
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

ctx = Context(agent)

async def run_async_code_d1e8bd50():
    logger.debug(await agent.run("What are the best resturants in toronto?", ctx=ctx))
    return 
 = asyncio.run(run_async_code_d1e8bd50())
logger.success(format_json())

async def run_async_code_f2bc1a40():
    logger.debug(await agent.run("tell me more about Osteria Giulia", ctx=ctx))
    return 
 = asyncio.run(run_async_code_f2bc1a40())
logger.success(format_json())

"""
## Avoiding Context Window Issues

The above example shows the core uses of the Metaphor tool. We can easily retrieve a clean list of links related to a query, and then we can fetch the content of the article as a cleaned up html extract. Alternatively, the search_and_retrieve_documents tool directly returns the documents from our search result.

We can see that the content of the articles is somewhat long compared to current LLM context windows, and so to allow retrieval and summary of many documents we will set up and use another tool from LlamaIndex that allows us to load text into a VectorStore, and query it for retrieval. This is where the `search_and_retrieve_documents` tool become particularly useful. The Agent can make a single query to retrieve a large number of documents, using a very small number of tokens, and then make queries to retrieve specific information from the documents.
"""
logger.info("## Avoiding Context Window Issues")


wrapped_retrieve = LoadAndSearchToolSpec.from_defaults(
    metaphor_tool_list[2],
)

"""
Our wrapped retrieval tools separate loading and reading into separate interfaces. We use `load` to load the documents into the vector store, and `read` to query the vector store. Let's try it out again
"""
logger.info("Our wrapped retrieval tools separate loading and reading into separate interfaces. We use `load` to load the documents into the vector store, and `read` to query the vector store. Let's try it out again")

wrapped_retrieve.load("This is the best explanation for machine learning transformers:")
logger.debug(wrapped_retrieve.read("what is a transformer"))
logger.debug(wrapped_retrieve.read("who wrote the first paper on transformers"))

"""
## Creating the Agent

We now are ready to create an Agent that can use Metaphors services to it's full potential. We will use our wrapped read and load tools, as well as the `get_date` utility for the following agent and test it out below:
"""
logger.info("## Creating the Agent")

agent = FunctionAgent(
    tools=[*wrapped_retrieve.to_tool_list(), metaphor_tool_list[4]],
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

logger.debug(
    await agent.run(
        "Can you summarize everything published in the last month regarding news on"
        " superconductors"
    )
)

"""
We asked the agent to retrieve documents related to superconductors from this month. It used the `get_date` tool to determine the current month, and then applied the filters in Metaphor based on publication date when calling `search`. It then loaded the documents using `retrieve_documents` and read them using `read_retrieve_documents`.

We can make another query to the vector store to read from it again, now that the articles are loaded:
"""
logger.info("We asked the agent to retrieve documents related to superconductors from this month. It used the `get_date` tool to determine the current month, and then applied the filters in Metaphor based on publication date when calling `search`. It then loaded the documents using `retrieve_documents` and read them using `read_retrieve_documents`.")

logger.info("\n\n[DONE]", bright=True)