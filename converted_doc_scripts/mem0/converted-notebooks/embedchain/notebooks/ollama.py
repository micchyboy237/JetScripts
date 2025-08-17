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
## Cookbook for using Ollama with Embedchain

### Step-1: Setup Ollama, follow these instructions https://github.com/jmorganca/ollama

Once Setup is done:

- ollama pull llama2 (All supported models can be found here: https://ollama.ai/library)
- ollama run llama2 (Test out the model once)
- ollama serve

### Step-2 Create embedchain app and define your config (all local inference)
"""
logger.info("## Cookbook for using Ollama with Embedchain")

app = App.from_config(config={
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama2",
            "temperature": 0.5,
            "top_p": 1,
            "stream": True
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5"
        }
    }
})

"""
### Step-3: Add data sources to your app
"""
logger.info("### Step-3: Add data sources to your app")

app.add("https://www.forbes.com/profile/elon-musk")

"""
### Step-4: All set. Now start asking questions related to your data
"""
logger.info("### Step-4: All set. Now start asking questions related to your data")

answer = app.query("who is elon musk?")

logger.info("\n\n[DONE]", bright=True)