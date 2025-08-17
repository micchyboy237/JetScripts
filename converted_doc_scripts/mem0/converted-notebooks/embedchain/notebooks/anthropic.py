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
## Cookbook for using Anthropic with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using Anthropic with Embedchain")

# !pip install embedchain

"""
### Step-2: Set Anthropic related environment variables

# You can find `OPENAI_API_KEY` on your [MLX dashboard](https://platform.openai.com/account/api-keys) and `ANTHROPIC_API_KEY` on your [Anthropic dashboard](https://console.anthropic.com/account/keys).
"""
logger.info("### Step-2: Set Anthropic related environment variables")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"
# os.environ["ANTHROPIC_API_KEY"] = "xxx"

"""
### Step-3: Create embedchain app and define your config
"""
logger.info("### Step-3: Create embedchain app and define your config")

app = App.from_config(config={
    "provider": "anthropic",
    "config": {
        "model": "claude-instant-1",
        "temperature": 0.5,
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