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
## Cookbook for using LLAMA2 with Embedchain

### Step-1: Install embedchain package
"""
logger.info("## Cookbook for using LLAMA2 with Embedchain")

# !pip install embedchain[llama2]

"""
### Step-2: Set LLAMA2 related environment variables

# You can find `OPENAI_API_KEY` on your [MLX dashboard](https://platform.openai.com/account/api-keys) and `REPLICATE_API_TOKEN` key on your [Replicate dashboard](https://replicate.com/account/api-tokens).
"""
logger.info("### Step-2: Set LLAMA2 related environment variables")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["REPLICATE_API_TOKEN"] = "xxx"

"""
### Step-3 Create embedchain app and define your config
"""
logger.info("### Step-3 Create embedchain app and define your config")

app = App.from_config(config={
    "provider": "llama2",
    "config": {
        "model": "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_p": 0.5,
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