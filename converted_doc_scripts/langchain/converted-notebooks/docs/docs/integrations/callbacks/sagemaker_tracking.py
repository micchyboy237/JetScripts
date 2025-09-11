from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import initialize_agent, load_tools
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.callbacks.sagemaker_callback import SageMakerCallbackHandler
from langchain_core.prompts import PromptTemplate
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.experiments.run import Run
from sagemaker.session import Session
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
# SageMaker Tracking

>[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that is used to quickly and easily build, train and deploy machine learning (ML) models. 

>[Amazon SageMaker Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html) is a capability of `Amazon SageMaker` that lets you organize, track, compare and evaluate ML experiments and model versions.

This notebook shows how LangChain Callback can be used to log and track prompts and other LLM hyperparameters into `SageMaker Experiments`. Here, we use different scenarios to showcase the capability:

* **Scenario 1**: *Single LLM* - A case where a single LLM model is used to generate output based on a given prompt.
* **Scenario 2**: *Sequential Chain* - A case where a sequential chain of two LLM models is used.
* **Scenario 3**: *Agent with Tools (Chain of Thought)* - A case where multiple tools (search and math) are used in addition to an LLM.


In this notebook, we will create a single experiment to log the prompts from each scenario.

## Installation and Setup
"""
logger.info("# SageMaker Tracking")

# %pip install --upgrade --quiet  sagemaker
# %pip install --upgrade --quiet  langchain-ollama
# %pip install --upgrade --quiet  google-search-results

"""
First, setup the required API keys

* Ollama: https://platform.ollama.com/account/api-keys (For Ollama LLM model)
* Google SERP API: https://serpapi.com/manage-api-key (For Google Search Tool)
"""
logger.info("First, setup the required API keys")


# os.environ["OPENAI_API_KEY"] = "<ADD-KEY-HERE>"
os.environ["SERPAPI_API_KEY"] = "<ADD-KEY-HERE>"



"""
## LLM Prompt Tracking
"""
logger.info("## LLM Prompt Tracking")

HPARAMS = {
    "temperature": 0.1,
    "model_name": "gpt-3.5-turbo-instruct",
}

BUCKET_NAME = None

EXPERIMENT_NAME = "langchain-sagemaker-tracker"

session = Session(default_bucket=BUCKET_NAME)

"""
### Scenario 1 - LLM
"""
logger.info("### Scenario 1 - LLM")

RUN_NAME = "run-scenario-1"
PROMPT_TEMPLATE = "tell me a joke about {topic}"
INPUT_VARIABLES = {"topic": "fish"}

with Run(
    experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME, sagemaker_session=session
) as run:
    sagemaker_callback = SageMakerCallbackHandler(run)

    llm = Ollama(callbacks=[sagemaker_callback], **HPARAMS)

    prompt = PromptTemplate.from_template(template=PROMPT_TEMPLATE)

    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[sagemaker_callback])

    chain.run(**INPUT_VARIABLES)

    sagemaker_callback.flush_tracker()

"""
### Scenario 2 - Sequential Chain
"""
logger.info("### Scenario 2 - Sequential Chain")

RUN_NAME = "run-scenario-2"

PROMPT_TEMPLATE_1 = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
Playwright: This is a synopsis for the above play:"""
PROMPT_TEMPLATE_2 = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
Play Synopsis: {synopsis}
Review from a New York Times play critic of the above play:"""

INPUT_VARIABLES = {
    "input": "documentary about good video games that push the boundary of game design"
}

with Run(
    experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME, sagemaker_session=session
) as run:
    sagemaker_callback = SageMakerCallbackHandler(run)

    prompt_template1 = PromptTemplate.from_template(template=PROMPT_TEMPLATE_1)
    prompt_template2 = PromptTemplate.from_template(template=PROMPT_TEMPLATE_2)

    llm = Ollama(callbacks=[sagemaker_callback], **HPARAMS)

    chain1 = LLMChain(llm=llm, prompt=prompt_template1, callbacks=[sagemaker_callback])

    chain2 = LLMChain(llm=llm, prompt=prompt_template2, callbacks=[sagemaker_callback])

    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2], callbacks=[sagemaker_callback]
    )

    overall_chain.run(**INPUT_VARIABLES)

    sagemaker_callback.flush_tracker()

"""
### Scenario 3 - Agent with Tools
"""
logger.info("### Scenario 3 - Agent with Tools")

RUN_NAME = "run-scenario-3"
PROMPT_TEMPLATE = "Who is the oldest person alive? And what is their current age raised to the power of 1.51?"

with Run(
    experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME, sagemaker_session=session
) as run:
    sagemaker_callback = SageMakerCallbackHandler(run)

    llm = Ollama(callbacks=[sagemaker_callback], **HPARAMS)

    tools = load_tools(["serpapi", "llm-math"], llm=llm, callbacks=[sagemaker_callback])

    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", callbacks=[sagemaker_callback]
    )

    agent.run(input=PROMPT_TEMPLATE)

    sagemaker_callback.flush_tracker()

"""
## Load Log Data

Once the prompts are logged, we can easily load and convert them to Pandas DataFrame as follows.
"""
logger.info("## Load Log Data")

logs = ExperimentAnalytics(experiment_name=EXPERIMENT_NAME)

df = logs.dataframe(force_refresh=True)

logger.debug(df.shape)
df.head()

"""
As can be seen above, there are three runs (rows) in the experiment corresponding to each scenario. Each run logs the prompts and related LLM settings/hyperparameters as json and are saved in s3 bucket. Feel free to load and explore the log data from each json path.
"""
logger.info("As can be seen above, there are three runs (rows) in the experiment corresponding to each scenario. Each run logs the prompts and related LLM settings/hyperparameters as json and are saved in s3 bucket. Feel free to load and explore the log data from each json path.")


logger.info("\n\n[DONE]", bright=True)