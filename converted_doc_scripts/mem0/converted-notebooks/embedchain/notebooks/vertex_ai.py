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
## Cookbook for using VertexAI with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using VertexAI with Embedchain")

# !pip install embedchain[vertexai]

"""
### Step-2: Set VertexAI related environment variables

# You can find `OPENAI_API_KEY` on your [MLX dashboard](https://platform.openai.com/account/api-keys).
"""
logger.info("### Step-2: Set VertexAI related environment variables")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

"""
### Step-3 Create embedchain app and define your config
"""
logger.info("### Step-3 Create embedchain app and define your config")

app = App.from_config(config={
    "llm": {
        "provider": "vertexai",
        "config": {
            "model": "chat-bison",
            "temperature": 0.5,
            "max_tokens": 1000,
            "stream": False
        }
    },
    "embedder": {
        "provider": "vertexai",
        "config": {
            "model": "textembedding-gecko"
        }
    }
})

"""
### Step-4: Add data sources to your app
"""
logger.info("### Step-4: Add data sources to your app")

app.add("https://www.forbes.com/profile/elon-musk")

"""
### Step-5: All set. Now start asking questions related to your data
"""
logger.info("### Step-5: All set. Now start asking questions related to your data")

while(True):
    question = input("Enter question: ")
    if question in ['q', 'exit', 'quit']:
        break
    answer = app.query(question)
    logger.debug(answer)

logger.info("\n\n[DONE]", bright=True)