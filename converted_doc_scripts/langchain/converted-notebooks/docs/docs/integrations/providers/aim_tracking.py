from datetime import datetime
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain_community.callbacks import AimCallbackHandler
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
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
# Aim

Aim makes it super easy to visualize and debug LangChain executions. Aim tracks inputs and outputs of LLMs and tools, as well as actions of agents. 

With Aim, you can easily debug and examine an individual execution:

![](https://user-images.githubusercontent.com/13848158/227784778-06b806c7-74a1-4d15-ab85-9ece09b458aa.png)

Additionally, you have the option to compare multiple executions side by side:

![](https://user-images.githubusercontent.com/13848158/227784994-699b24b7-e69b-48f9-9ffa-e6a6142fd719.png)

Aim is fully open source, [learn more](https://github.com/aimhubio/aim) about Aim on GitHub.

Let's move forward and see how to enable and configure Aim callback.

<h3>Tracking LangChain Executions with Aim</h3>

In this notebook we will explore three usage scenarios. To start off, we will install the necessary packages and import certain modules. Subsequently, we will configure two environment variables that can be established either within the Python script or through the terminal.
"""
logger.info("# Aim")

# %pip install --upgrade --quiet  aim
# %pip install --upgrade --quiet  langchain
# %pip install --upgrade --quiet  langchain-ollama
# %pip install --upgrade --quiet  google-search-results


"""
Our examples use a GPT model as the LLM, and Ollama offers an API for this purpose. You can obtain the key from the following link: https://platform.ollama.com/account/api-keys .

We will use the SerpApi to retrieve search results from Google. To acquire the SerpApi key, please go to https://serpapi.com/manage-api-key .
"""
logger.info("Our examples use a GPT model as the LLM, and Ollama offers an API for this purpose. You can obtain the key from the following link: https://platform.ollama.com/account/api-keys .")

# os.environ["OPENAI_API_KEY"] = "..."
os.environ["SERPAPI_API_KEY"] = "..."

"""
The event methods of `AimCallbackHandler` accept the LangChain module or agent as input and log at least the prompts and generated results, as well as the serialized version of the LangChain module, to the designated Aim run.
"""
logger.info("The event methods of `AimCallbackHandler` accept the LangChain module or agent as input and log at least the prompts and generated results, as well as the serialized version of the LangChain module, to the designated Aim run.")

session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
aim_callback = AimCallbackHandler(
    repo=".",
    experiment_name="scenario 1: Ollama LLM",
)

callbacks = [StdOutCallbackHandler(), aim_callback]
llm = ChatOllama(temperature=0, callbacks=callbacks)

"""
The `flush_tracker` function is used to record LangChain assets on Aim. By default, the session is reset rather than being terminated outright.

<h3>Scenario 1</h3> In the first scenario, we will use Ollama LLM.
"""
logger.info("The `flush_tracker` function is used to record LangChain assets on Aim. By default, the session is reset rather than being terminated outright.")

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 3)
aim_callback.flush_tracker(
    langchain_asset=llm,
    experiment_name="scenario 2: Chain with multiple SubChains on multiple generations",
)

"""
<h3>Scenario 2</h3> Scenario two involves chaining with multiple SubChains across multiple generations.
"""


template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, callbacks=callbacks)

test_prompts = [
    {
        "title": "documentary about good video games that push the boundary of game design"
    },
    {"title": "the phenomenon behind the remarkable speed of cheetahs"},
    {"title": "the best in class mlops tooling"},
]
synopsis_chain.apply(test_prompts)
aim_callback.flush_tracker(
    langchain_asset=synopsis_chain, experiment_name="scenario 3: Agent with Tools"
)

"""
<h3>Scenario 3</h3> The third scenario involves an agent with tools.
"""


tools = load_tools(["serpapi", "llm-math"], llm=llm, callbacks=callbacks)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=callbacks,
)
agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)
aim_callback.flush_tracker(langchain_asset=agent, reset=False, finish=True)

logger.info("\n\n[DONE]", bright=True)
