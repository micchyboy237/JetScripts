from jet.logger import logger
from langchain.chains import LLMChain, PromptTemplate
from langchain_community.chat_models import ChatMlflow
from langchain_community.embeddings import MlflowEmbeddings
from langchain_community.llms import Mlflow
from langchain_core.messages import HumanMessage, SystemMessage
import mlflow
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
# MLflow AI Gateway for LLMs

>[The MLflow AI Gateway for LLMs](https://www.mlflow.org/docs/latest/llms/deployments/index.html) is a powerful tool designed to streamline the usage and management of various large
> language model (LLM) providers, such as Ollama and Ollama, within an organization. It offers a high-level interface
> that simplifies the interaction with these services by providing a unified endpoint to handle specific LLM related requests.

## Installation and Setup

Install `mlflow` with MLflow GenAI dependencies:
"""
logger.info("# MLflow AI Gateway for LLMs")

pip install 'mlflow[genai]'

"""
Set the Ollama API key as an environment variable:
"""
logger.info("Set the Ollama API key as an environment variable:")

# export OPENAI_API_KEY=...

"""
Create a configuration file:
"""
logger.info("Create a configuration file:")

endpoints:
  - name: completions
    endpoint_type: llm/v1/completions
    model:
      provider: ollama
      name: text-davinci-003
      config:
#         ollama_api_key: $OPENAI_API_KEY

  - name: embeddings
    endpoint_type: llm/v1/embeddings
    model:
      provider: ollama
      name: text-embedding-ada-002
      config:
#         ollama_api_key: $OPENAI_API_KEY

"""
Start the gateway server:
"""
logger.info("Start the gateway server:")

mlflow gateway start --config-path /path/to/config.yaml

"""
## Example provided by `MLflow`

>The `mlflow.langchain` module provides an API for logging and loading `LangChain` models.
> This module exports multivariate LangChain models in the langchain flavor and univariate LangChain
> models in the pyfunc flavor.

See the [API documentation and examples](https://www.mlflow.org/docs/latest/llms/langchain/index.html) for more information.

## Completions Example
"""
logger.info("## Example provided by `MLflow`")


llm = Mlflow(
    target_uri="http://127.0.0.1:5000",
    endpoint="completions",
)

llm_chain = LLMChain(
    llm=Mlflow,
    prompt=PromptTemplate(
        input_variables=["adjective"],
        template="Tell me a {adjective} joke",
    ),
)
result = llm_chain.run(adjective="funny")
logger.debug(result)

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(chain, "model")

model = mlflow.pyfunc.load_model(model_info.model_uri)
logger.debug(model.predict([{"adjective": "funny"}]))

"""
## Embeddings Example
"""
logger.info("## Embeddings Example")


embeddings = MlflowEmbeddings(
    target_uri="http://127.0.0.1:5000",
    endpoint="embeddings",
)

logger.debug(embeddings.embed_query("hello"))
logger.debug(embeddings.embed_documents(["hello"]))

"""
## Chat Example
"""
logger.info("## Chat Example")


chat = ChatMlflow(
    target_uri="http://127.0.0.1:5000",
    endpoint="chat",
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French: I love programming."
    ),
]
logger.debug(chat(messages))

logger.info("\n\n[DONE]", bright=True)