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
## Cookbook for using JinaChat with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using JinaChat with Embedchain")

# !pip install embedchain

"""
### Step-2: Set JinaChat related environment variables

# You can find `OPENAI_API_KEY` on your [MLX dashboard](https://platform.openai.com/account/api-keys) and `JINACHAT_API_KEY` key on your [Chat Jina dashboard](https://chat.jina.ai/api).
"""
logger.info("### Step-2: Set JinaChat related environment variables")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["JINACHAT_API_KEY"] = "xxx"

"""
### Step-3 Create embedchain app and define your config
"""
logger.info("### Step-3 Create embedchain app and define your config")

app = App.from_config(config={
    "provider": "jina",
    "config": {
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_p": 1,
        "stream": False
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