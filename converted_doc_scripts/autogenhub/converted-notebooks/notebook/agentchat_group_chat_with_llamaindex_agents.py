from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.tools.wikipedia import WikipediaToolSpec
import autogen
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Groupchat with Llamaindex agents

[Llamaindex agents](https://docs.llamaindex.ai/en/stable/optimizing/agentic_strategies/agentic_strategies/) have the ability to use planning strategies to answer user questions. They can be integrated in Autogen in easy ways

## Requirements
"""
logger.info("# Groupchat with Llamaindex agents")

# %pip install pyautogen llama-index llama-index-tools-wikipedia llama-index-readers-wikipedia wikipedia

"""
## Set your API Endpoint
"""
logger.info("## Set your API Endpoint")



config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"tags": ["gpt-3.5-turbo"]},  # comment out to get all
)

"""
## Set Llamaindex
"""
logger.info("## Set Llamaindex")


llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096,
    temperature=0.0,
    api_key=os.environ.get("OPENAPI_API_KEY", ""),
)

embed_model = OllamaEmbedding(
    model="text-embedding-ada-002",
    temperature=0.0,
    api_key=os.environ.get("OPENAPI_API_KEY", ""),
)

Settings.llm = llm
Settings.embed_model = embed_model

wiki_spec = WikipediaToolSpec()
wikipedia_tool = wiki_spec.to_tool_list()[1]

location_specialist = ReActAgent.from_tools(tools=[wikipedia_tool], llm=llm, max_iterations=10, verbose=True)

"""
## Create agents

In this example, we will create a Llamaindex agent to answer questions fecting data from wikipedia and a user proxy agent.
"""
logger.info("## Create agents")


llm_config = {
    "temperature": 0,
    "config_list": config_list,
}

trip_assistant = LLamaIndexConversableAgent(
    "trip_specialist",
    llama_index_agent=location_specialist,
    system_message="You help customers finding more about places they would like to visit. You can use external resources to provide more details as you engage with the customer.",
    description="This agents helps customers discover locations to visit, things to do, and other details about a location. It can use external resources to provide more details. This agent helps in finding attractions, history and all that there si to know about a place",
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="ALWAYS",
    code_execution_config=False,
)

"""
Next, let's set up our group chat.
"""
logger.info("Next, let's set up our group chat.")

groupchat = autogen.GroupChat(
    agents=[trip_assistant, user_proxy],
    messages=[],
    max_round=500,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

chat_result = user_proxy.initiate_chat(
    manager,
    message="""
What can i find in Tokyo related to Hayao Miyazaki and its moveis like Spirited Away?.
""",
)

logger.info("\n\n[DONE]", bright=True)