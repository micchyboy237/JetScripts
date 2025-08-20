from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.gmail.base import GmailToolSpec
from llama_index.tools.google_calendar.base import GoogleCalendarToolSpec
from llama_index.tools.google_search.base import GoogleSearchToolSpec
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
# Google combined tools example

This notebook features a more advanced usage of Agent Tools, using the Google Calendar, Mail and Search integrations as well as the Load and Search Meta Tool to fufill a more complicated set of tasks for the user.

## Setup the Tools
First we will import MLX and setup the Agent:
"""
logger.info("# Google combined tools example")


# os.environ["OPENAI_API_KEY"] = "sk-api-key"


"""
Now we can import the Google Tools we are going to use. See the README for the respective tools to get started with authentication.
"""
logger.info("Now we can import the Google Tools we are going to use. See the README for the respective tools to get started with authentication.")


gmail_tools = GmailToolSpec().to_tool_list()
gcal_tools = GoogleCalendarToolSpec().to_tool_list()
gsearch_tools = GoogleSearchToolSpec(key="api-key", engine="engine").to_tool_list()

"""
Let's take a look at all of the tools we have available from the 3 tool specs we initialized:
"""
logger.info("Let's take a look at all of the tools we have available from the 3 tool specs we initialized:")

for tool in [*gmail_tools, *gcal_tools, *gsearch_tools]:
    logger.debug(tool.metadata.name)
    logger.debug(tool.metadata.description)

"""
We have to be conscious of the models context length when using these tools as if we are not careful the response can easily be larger than the token limit. In particular, the load_data function for emails returns large payloads, as well as google search. In this example I will wrap those two tools in the Load and Search Meta tool:
"""
logger.info("We have to be conscious of the models context length when using these tools as if we are not careful the response can easily be larger than the token limit. In particular, the load_data function for emails returns large payloads, as well as google search. In this example I will wrap those two tools in the Load and Search Meta tool:")


logger.debug("Wrapping " + gsearch_tools[0].metadata.name)
gsearch_load_and_search_tools = LoadAndSearchToolSpec.from_defaults(
    gsearch_tools[0],
).to_tool_list()

logger.debug("Wrapping gmail " + gmail_tools[0].metadata.name)
gmail_load_and_search_tools = LoadAndSearchToolSpec.from_defaults(
    gmail_tools[0],
).to_tool_list()

logger.debug("Wrapping google calendar " + gcal_tools[0].metadata.name)
gcal_load_and_search_tools = LoadAndSearchToolSpec.from_defaults(
    gcal_tools[0],
).to_tool_list()

"""
Notice we are only wrapping individual tools out of the tool list. Lets combine the all the tools together into a combined list:
"""
logger.info("Notice we are only wrapping individual tools out of the tool list. Lets combine the all the tools together into a combined list:")

all_tools = [
    *gsearch_load_and_search_tools,
    *gmail_load_and_search_tools,
    *gcal_load_and_search_tools,
    *gcal_tools[1::],
    *gmail_tools[1::],
    *gsearch_tools[1::],
]

"""
Now the tools are ready to pass to the agent:
"""
logger.info("Now the tools are ready to pass to the agent:")

agent = FunctionAgent(tools=all_tools, llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"))

"""
## Interacting with the Agent

We are now ready to interact with the Agent and test the calendar, email and search capabilities! Let's try out the search first:
"""
logger.info("## Interacting with the Agent")

await agent.run(
    "search google and find the email address for a dentist in toronto near bloor and"
    " dufferin"
)

"""
## Summary

We were able to use the google search, email and calendar tools to find a dentist and request an appointment, making sure any calendar conflicts were avoided. This notebook should prove useful for experimenting with more complicated agents, combining together the tools available in LlamaHub to create more complicated workflows the agent can execute.
"""
logger.info("## Summary")

logger.info("\n\n[DONE]", bright=True)