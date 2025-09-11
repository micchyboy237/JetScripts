from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from ray import serve
from starlette.requests import Request
import os
import requests
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
# Ray Serve

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is a scalable model serving library for building online inference APIs. Serve is particularly well suited for system composition, enabling you to build a complex inference service consisting of multiple chains and business logic all in Python code.

## Goal of this notebook
This notebook shows a simple example of how to deploy an Ollama chain into production. You can extend it to deploy your own self-hosted models where you can easily define amount of hardware resources (GPUs and CPUs) needed to run your model in production efficiently. Read more about available options including autoscaling in the Ray Serve [documentation](https://docs.ray.io/en/latest/serve/getting_started.html).

## Setup Ray Serve
Install ray with `pip install ray[serve]`.

## General Skeleton

The general skeleton for deploying a service is the following:
"""
logger.info("# Ray Serve")


@serve.deployment
class LLMServe:
    def __init__(self) -> None:
        pass

    async def __call__(self, request: Request) -> str:
        return "Hello World"


deployment = LLMServe.bind()

serve.api.run(deployment)

serve.api.shutdown()

"""
## Example of deploying and Ollama chain with custom prompts

Get an Ollama API key from [here](https://platform.ollama.com/account/api-keys). By running the following code, you will be asked to provide your API key.
"""
logger.info("## Example of deploying and Ollama chain with custom prompts")


# from getpass import getpass

# OPENAI_API_KEY = getpass()

@serve.deployment
class DeployLLM:
    def __init__(self):
        #         llm = ChatOllama(ollama_api_key=OPENAI_API_KEY)
        template = "Question: {question}\n\nAnswer: Let's think step by step."
        prompt = PromptTemplate.from_template(template)
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def _run_chain(self, text: str):
        return self.chain(text)

    async def __call__(self, request: Request):
        text = request.query_params["text"]
        resp = self._run_chain(text)
        return resp["text"]


"""
Now we can bind the deployment.
"""
logger.info("Now we can bind the deployment.")

deployment = DeployLLM.bind()

"""
We can assign the port number and host when we want to run the deployment.
"""
logger.info(
    "We can assign the port number and host when we want to run the deployment.")

PORT_NUMBER = 8282
serve.api.run(deployment, port=PORT_NUMBER)

"""
Now that service is deployed on port `localhost:8282` we can send a post request to get the results back.
"""
logger.info("Now that service is deployed on port `localhost:8282` we can send a post request to get the results back.")


text = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = requests.post(f"http://localhost:{PORT_NUMBER}/?text={text}")
logger.debug(response.content.decode())

logger.info("\n\n[DONE]", bright=True)
