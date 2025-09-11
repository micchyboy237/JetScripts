from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import Clarifai
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
# Clarifai

>[Clarifai](https://www.clarifai.com/) is an AI Platform that provides the full AI lifecycle ranging from data exploration, data labeling, model training, evaluation, and inference.

This example goes over how to use LangChain to interact with `Clarifai` [models](https://clarifai.com/explore/models). 

To use Clarifai, you must have an account and a Personal Access Token (PAT) key. 
[Check here](https://clarifai.com/settings/security) to get or create a PAT.

# Dependencies
"""
logger.info("# Clarifai")

# %pip install --upgrade --quiet  clarifai


os.environ["CLARIFAI_PAT"] = "CLARIFAI_PAT_TOKEN"

"""
# Imports
Here we will be setting the personal access token. You can find your PAT under [settings/security](https://clarifai.com/settings/security) in your Clarifai account.
"""
logger.info("# Imports")

# from getpass import getpass

# CLARIFAI_PAT = getpass()


"""
# Input
Create a prompt template to be used with the LLM Chain:
"""
logger.info("# Input")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

"""
# Setup
Setup the user id and app id where the model resides. You can find a list of public models on https://clarifai.com/explore/models

You will have to also initialize the model id and if needed, the model version id. Some models have many versions, you can choose the one appropriate for your task.
                                                              
Alternatively, You can use the model_url (for ex: "https://clarifai.com/anthropic/completion/models/claude-v2") for initialization.
"""
logger.info("# Setup")

USER_ID = "ollama"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"


MODEL_URL = "https://clarifai.com/ollama/chat-completion/models/GPT-4"

clarifai_llm = Clarifai(user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
clarifai_llm = Clarifai(model_url=MODEL_URL)

llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)

"""
# Run Chain
"""
logger.info("# Run Chain")

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

"""
## Model Predict with Inference parameters for GPT.
Alternatively you can use GPT models with inference parameters (like temperature, max_tokens etc)
"""
logger.info("## Model Predict with Inference parameters for GPT.")

params = dict(temperature=str(0.3), max_tokens=100)

clarifai_llm = Clarifai(user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
llm_chain = LLMChain(
    prompt=prompt, llm=clarifai_llm, llm_kwargs={"inference_params": params}
)

question = "How many 3 digit even numbers you can form that if one of the digits is 5 then the following digit must be 7?"

llm_chain.run(question)

"""
Generate responses for list of prompts
"""
logger.info("Generate responses for list of prompts")

clarifai_llm._generate(
    [
        "Help me summarize the events of american revolution in 5 sentences",
        "Explain about rocket science in a funny way",
        "Create a script for welcome speech for the college sports day",
    ],
    inference_params=params,
)

logger.info("\n\n[DONE]", bright=True)