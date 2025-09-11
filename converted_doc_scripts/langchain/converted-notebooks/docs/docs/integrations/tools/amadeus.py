from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain_community.llms import HuggingFaceHub
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
# Amadeus Toolkit

This notebook walks you through connecting LangChain to the `Amadeus` travel APIs.

This `Amadeus` toolkit allows agents to make decision when it comes to travel, especially searching and booking trips with flights.

To use this toolkit, you will need to have your Amadeus API keys ready, explained in the [Get started Amadeus Self-Service APIs](https://developers.amadeus.com/get-started/get-started-with-self-service-apis-335). Once you've received a AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET, you can input them as environmental variables below.

Note: Amadeus Self-Service APIs offers a test environment with [free limited data](https://amadeus4dev.github.io/developer-guides/test-data/). This allows developers to build and test their applications before deploying them to production. To access real-time data, you will need to [move to the production environment](https://amadeus4dev.github.io/developer-guides/API-Keys/moving-to-production/).
"""
logger.info("# Amadeus Toolkit")

# %pip install --upgrade --quiet  amadeus > /dev/null

# %pip install -qU langchain-community

"""
## Assign Environmental Variables

The toolkit will read the AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environmental variables to authenticate the user, so you need to set them here.
"""
logger.info("## Assign Environmental Variables")


os.environ["AMADEUS_CLIENT_ID"] = "CLIENT_ID"
os.environ["AMADEUS_CLIENT_SECRET"] = "CLIENT_SECRET"

"""
## Create the Amadeus Toolkit and Get Tools

To start, you need to create the toolkit, so you can access its tools later.

# By default, `AmadeusToolkit` uses `ChatOllama` to identify airports closest to a given location. To use it, just set `OPENAI_API_KEY`.
"""
logger.info("## Create the Amadeus Toolkit and Get Tools")

# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


toolkit = AmadeusToolkit()
tools = toolkit.get_tools()

"""
Alternatively, you can use any LLM supported by langchain, e.g. `HuggingFaceHub`.
"""
logger.info("Alternatively, you can use any LLM supported by langchain, e.g. `HuggingFaceHub`.")


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_API_TOKEN"

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_length": 64},
)

toolkit_hf = AmadeusToolkit(llm=llm)

"""
## Use Amadeus Toolkit within an Agent
"""
logger.info("## Use Amadeus Toolkit within an Agent")


llm = ChatOllama(model="llama3.2")

prompt = hub.pull("hwchase17/react-json")
agent = create_react_agent(
    llm,
    tools,
    prompt,
    tools_renderer=render_text_description_and_args,
    output_parser=ReActJsonSingleInputOutputParser(),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "What is the name of the airport in Cali, Colombia?"})

agent_executor.invoke(
    {
        "input": "What is the departure time of the cheapest flight on March 10, 2024 leaving Dallas, Texas before noon to Lincoln, Nebraska?"
    }
)

agent_executor.invoke(
    {
        "input": "At what time does earliest flight on March 10, 2024 leaving Dallas, Texas to Lincoln, Nebraska land in Nebraska?"
    }
)

agent_executor.invoke(
    {
        "input": "What is the full travel time for the cheapest flight between Portland, Oregon to Dallas, TX on March 10, 2024?"
    }
)

agent_executor.invoke(
    {
        "input": "Please draft a concise email from Santiago to Paul, Santiago's travel agent, asking him to book the earliest flight from DFW to DCA on March 10, 2024. Include all flight details in the email."
    }
)

logger.info("\n\n[DONE]", bright=True)