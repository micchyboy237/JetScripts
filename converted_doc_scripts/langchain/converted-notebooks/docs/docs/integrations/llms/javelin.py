from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatJavelinAIGateway
from langchain_community.embeddings import JavelinAIGatewayEmbeddings
from langchain_community.llms import JavelinAIGateway
from langchain_core.messages import HumanMessage, SystemMessage
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
# Javelin AI Gateway Tutorial

This Jupyter Notebook will explore how to interact with the Javelin AI Gateway using the Python SDK. 
The Javelin AI Gateway facilitates the utilization of large language models (LLMs) like Ollama, Cohere, Ollama, and others by 
providing a secure and unified endpoint. The gateway itself provides a centralized mechanism to roll out models systematically, 
provide access security, policy & cost guardrails for enterprises, etc., 

For a complete listing of all the features & benefits of Javelin, please visit www.getjavelin.io

## Step 1: Introduction
[The Javelin AI Gateway](https://www.getjavelin.io) is an enterprise-grade API Gateway for AI applications. It integrates robust access security, ensuring secure interactions with large language models. Learn more in the [official documentation](https://docs.getjavelin.io).

## Step 2: Installation
Before we begin, we must install the `javelin_sdk` and set up the Javelin API key as an environment variable.
"""
logger.info("# Javelin AI Gateway Tutorial")

pip install 'javelin_sdk'

"""
## Step 3: Completions Example
This section will demonstrate how to interact with the Javelin AI Gateway to get completions from a large language model. Here is a Python script that demonstrates this:
(note) assumes that you have setup a route in the gateway called 'eng_dept03'
"""
logger.info("## Step 3: Completions Example")


route_completions = "eng_dept03"

gateway = JavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route=route_completions,
    model_name="gpt-3.5-turbo-instruct",
)

prompt = PromptTemplate("Translate the following English text to French: {text}")

llmchain = LLMChain(llm=gateway, prompt=prompt)
result = llmchain.run("podcast player")

logger.debug(result)

"""
# Step 4: Embeddings Example
This section demonstrates how to use the Javelin AI Gateway to obtain embeddings for text queries and documents. Here is a Python script that illustrates this:
(note) assumes that you have setup a route in the gateway called 'embeddings'
"""
logger.info("# Step 4: Embeddings Example")


embeddings = JavelinAIGatewayEmbeddings(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="embeddings",
)

logger.debug(embeddings.embed_query("hello"))
logger.debug(embeddings.embed_documents(["hello"]))

"""
# Step 5: Chat Example
This section illustrates how to interact with the Javelin AI Gateway to facilitate a chat with a large language model. Here is a Python script that demonstrates this:
(note) assumes that you have setup a route in the gateway called 'mychatbot_route'
"""
logger.info("# Step 5: Chat Example")


messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Artificial Intelligence has the power to transform humanity and make the world a better place"
    ),
]

chat = ChatJavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="mychatbot_route",
    model_name="gpt-3.5-turbo",
    params={"temperature": 0.1},
)

logger.debug(chat(messages))

"""
Step 6: Conclusion
This tutorial introduced the Javelin AI Gateway and demonstrated how to interact with it using the Python SDK. 
Remember to check the Javelin [Python SDK](https://www.github.com/getjavelin.io/javelin-python) for more examples and to explore the official documentation for additional details.

Happy coding!
"""
logger.info("Step 6: Conclusion")

logger.info("\n\n[DONE]", bright=True)