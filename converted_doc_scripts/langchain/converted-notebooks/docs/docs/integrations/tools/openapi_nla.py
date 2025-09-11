from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import NLAToolkit
from langchain_community.utilities import Requests
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
# Natural Language API Toolkits

`Natural Language API` Toolkits (`NLAToolkits`) permit LangChain Agents to efficiently plan and combine calls across endpoints. 

This notebook demonstrates a sample composition of the `Speak`, `Klarna`, and `Spoonacluar` APIs.

### First, import dependencies and load the LLM
"""
logger.info("# Natural Language API Toolkits")


llm = ChatOllama(
    temperature=0, max_tokens=700, model_name="gpt-3.5-turbo-instruct"
)  # You can swap between different core LLM's here.

"""
### Next, load the Natural Language API Toolkits
"""
logger.info("### Next, load the Natural Language API Toolkits")

speak_toolkit = NLAToolkit.from_llm_and_url(
    llm, "https://api.speak.com/openapi.yaml")
klarna_toolkit = NLAToolkit.from_llm_and_url(
    llm, "https://www.klarna.com/us/shopping/public/ollama/v0/api-docs/"
)

"""
### Create the Agent
"""
logger.info("### Create the Agent")

openapi_format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action representative.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
Final Answer: the final answer to the original input question with the right amount of detail

When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""

natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()
mrkl = initialize_agent(
    natural_language_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"format_instructions": openapi_format_instructions},
)

mrkl.run(
    "I have an end of year party for my Italian class and have to buy some Italian clothes for it"
)

"""
### Use Auth and add more Endpoints

Some endpoints may require user authentication via things like access tokens. Here we show how to pass in the authentication information via the `Requests` wrapper object.

Since each NLATool exposes a concisee natural language interface to its wrapped API, the top level conversational agent has an easier job incorporating each endpoint to satisfy a user's request.

**Adding the Spoonacular endpoints.**

1. Go to the [Spoonacular API Console](https://spoonacular.com/food-api/console#Profile) and make a free account.
2. Click on `Profile` and copy your API key below.
"""
logger.info("### Use Auth and add more Endpoints")

spoonacular_  # Copy from the API Console

requests = Requests(headers={"x-api-key": spoonacular_api_key})
spoonacular_toolkit = NLAToolkit.from_llm_and_url(
    llm,
    "https://spoonacular.com/application/frontend/downloads/spoonacular-openapi-3.json",
    requests=requests,
    max_text_length=1800,  # If you want to truncate the response text
)

natural_language_api_tools = (
    speak_toolkit.get_tools()
    + klarna_toolkit.get_tools()
    + spoonacular_toolkit.get_tools()[:30]
)
logger.debug(f"{len(natural_language_api_tools)} tools loaded.")

mrkl = initialize_agent(
    natural_language_api_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"format_instructions": openapi_format_instructions},
)

user_input = (
    "I'm learning Italian, and my language class is having an end of year party... "
    " Could you help me find an Italian outfit to wear and"
    " an appropriate recipe to prepare so I can present for the class in Italian?"
)

mrkl.run(user_input)

"""
## Thank you!
"""
logger.info("## Thank you!")

natural_language_api_tools[1].run(
    "Tell the LangChain audience to 'enjoy the meal' in Italian, please!"
)

logger.info("\n\n[DONE]", bright=True)
