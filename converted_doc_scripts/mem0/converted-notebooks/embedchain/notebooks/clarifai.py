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
# Cookbook for using Clarifai LLM and Embedders with Embedchain

## Step-1: Install embedchain-clarifai package
"""
logger.info("# Cookbook for using Clarifai LLM and Embedders with Embedchain")

# !pip install embedchain[clarifai]

"""
## Step-2: Set Clarifai PAT as env variable.
Sign-up to [Clarifai](https://clarifai.com/signup?utm_source=clarifai_home&utm_medium=direct&) platform and you can obtain `CLARIFAI_PAT` by following this [link](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/).

optionally you can also pass `api_key` in config of llm/embedder class.
"""
logger.info("## Step-2: Set Clarifai PAT as env variable.")


os.environ["CLARIFAI_PAT"]="xxx"

"""
## Step-3 Create embedchain app using clarifai LLM and embedder and define your config.

Browse through Clarifai community page to get the URL of different [LLM](https://clarifai.com/explore/models?page=1&perPage=24&filterData=%5B%7B%22field%22%3A%22use_cases%22%2C%22value%22%3A%5B%22llm%22%5D%7D%5D) and [embedding](https://clarifai.com/explore/models?page=1&perPage=24&filterData=%5B%7B%22field%22%3A%22input_fields%22%2C%22value%22%3A%5B%22text%22%5D%7D%2C%7B%22field%22%3A%22output_fields%22%2C%22value%22%3A%5B%22embeddings%22%5D%7D%5D) models available.
"""
logger.info("## Step-3 Create embedchain app using clarifai LLM and embedder and define your config.")

app = App.from_config(config={
    "llm": {
        "provider": "clarifai",
        "config": {
            "model": "https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct",
            "model_kwargs": {
            "temperature": 0.5,
            "max_tokens": 1000
            }
        }
    },
    "embedder": {
        "provider": "clarifai",
        "config": {
            "model": "https://clarifai.com/openai/embed/models/text-embedding-ada",
        }
}
})

"""
## Step-4: Add data sources to your app
"""
logger.info("## Step-4: Add data sources to your app")

app.add("https://www.forbes.com/profile/elon-musk")

"""
## Step-5: All set. Now start asking questions related to your data
"""
logger.info("## Step-5: All set. Now start asking questions related to your data")

while(True):
    question = input("Enter question: ")
    if question in ['q', 'exit', 'quit']:
        break
    answer = app.query(question)
    logger.debug(answer)

logger.info("\n\n[DONE]", bright=True)