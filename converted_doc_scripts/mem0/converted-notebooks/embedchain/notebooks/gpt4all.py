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
## Cookbook for using GPT4All with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using GPT4All with Embedchain")

# !pip install embedchain[opensource]

"""
### Step-2: Set GPT4ALL related environment variables

GPT4All is free for all and doesn't require any API Key to use it. So you can use it for free!
"""
logger.info("### Step-2: Set GPT4ALL related environment variables")


"""
### Step-3 Create embedchain app and define your config
"""
logger.info("### Step-3 Create embedchain app and define your config")

app = App.from_config(config={
    "llm": {
        "provider": "gpt4all",
        "config": {
            "model": "orca-mini-3b-gguf2-q4_0.gguf",
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 1,
            "stream": False
        }
    },
    "embedder": {
        "provider": "gpt4all",
        "config": {
            "model": "all-MiniLM-L6-v2"
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