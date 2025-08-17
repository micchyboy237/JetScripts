from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Cookbook for using Azure MLX with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using Azure MLX with Embedchain")

# !pip install embedchain

"""
### Step-2: Set Azure MLX related environment variables

You can find these env variables on your Azure MLX dashboard.
"""
logger.info("### Step-2: Set Azure MLX related environment variables")


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://xxx.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_API_VERSION"] = "xxx"

"""
### Step-3: Define your llm and embedding model config
"""
logger.info("### Step-3: Define your llm and embedding model config")

config = """
llm:
  provider: azure_openai
  model: gpt-35-turbo
  config:
    deployment_name: ec_openai_azure
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

embedder:
  provider: azure_openai
  config:
    model: text-embedding-ada-002
    deployment_name: ec_embeddings_ada_002
"""

with open('azure_openai.yaml', 'w') as file:
    file.write(config)

"""
### Step-4 Create embedchain app based on the config
"""
logger.info("### Step-4 Create embedchain app based on the config")

app = App.from_config(config_path="azure_openai.yaml")

"""
### Step-5: Add data sources to your app
"""
logger.info("### Step-5: Add data sources to your app")

app.add("https://www.forbes.com/profile/elon-musk")

"""
### Step-6: All set. Now start asking questions related to your data
"""
logger.info("### Step-6: All set. Now start asking questions related to your data")

while(True):
    question = input("Enter question: ")
    if question in ['q', 'exit', 'quit']:
        break
    answer = app.query(question)
    logger.debug(answer)

logger.info("\n\n[DONE]", bright=True)