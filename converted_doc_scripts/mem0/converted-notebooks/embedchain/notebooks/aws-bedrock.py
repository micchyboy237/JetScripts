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
### Step-2: Set AWS related environment variables

You can find these env variables on your AWS Management Console.
"""
logger.info("### Step-2: Set AWS related environment variables")


os.environ["AWS_ACCESS_KEY_ID"] = "AKIAIOSFODNN7EXAMPLE" # replace with your AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" # replace with your AWS_SECRET_ACCESS_KEY
os.environ["AWS_SESSION_TOKEN"] = "IQoJb3JpZ2luX2VjEJr...==" # replace with your AWS_SESSION_TOKEN
os.environ["AWS_DEFAULT_REGION"] = "us-east-1" # replace with your AWS_DEFAULT_REGION


"""
### Step-3: Define your llm and embedding model config

May need to install langchain-anthropic to try with claude models
"""
logger.info("### Step-3: Define your llm and embedding model config")

config = """
llm:
  provider: aws_bedrock
  config:
    model: 'amazon.titan-text-express-v1'
    deployment_name: ec_titan_express_v1
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

embedder:
  provider: aws_bedrock
  config:
    model: amazon.titan-embed-text-v2:0
    deployment_name: ec_embeddings_titan_v2
"""

with open('aws_bedrock.yaml', 'w') as file:
    file.write(config)

"""
### Step-4 Create two embedchain apps based on the config
"""
logger.info("### Step-4 Create two embedchain apps based on the config")

app = App.from_config(config_path="aws_bedrock.yaml")
app.reset() # Reset the app to clear the cache and start fresh

"""
### Step-5: Add a data source to unrelated to the question you are asking
"""
logger.info("### Step-5: Add a data source to unrelated to the question you are asking")

app.add("https://www.lipsum.com/")

"""
### Step-6: Notice the underlying context changing with the updated data source
"""
logger.info("### Step-6: Notice the underlying context changing with the updated data source")

question = "Who is Elon Musk?"
context = " ".join([a['context'] for a in app.search(question)])
logger.debug("Context:", context)
app.add("https://www.forbes.com/profile/elon-musk")
context = " ".join([a['context'] for a in app.search(question)])
logger.debug("Context with updated memory:", context)

logger.info("\n\n[DONE]", bright=True)