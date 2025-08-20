from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.clarifai import Clarifai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/clarifai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Clarifai LLM

## Example notebook to call different LLM models using Clarifai

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Clarifai LLM")

# %pip install llama-index-llms-clarifai

# !pip install llama-index

"""
Install clarifai
"""
logger.info("Install clarifai")

# !pip install clarifai

"""
Set clarifai PAT as environment variable.
"""
logger.info("Set clarifai PAT as environment variable.")


os.environ["CLARIFAI_PAT"] = "<YOUR CLARIFAI PAT>"

"""
Import clarifai package
"""
logger.info("Import clarifai package")


"""
Explore various models according to your prefrence from
[Our Models page](https://clarifai.com/explore/models?filterData=%5B%7B%22field%22%3A%22use_cases%22%2C%22value%22%3A%5B%22llm%22%5D%7D%5D&page=2&perPage=24)
"""
logger.info("Explore various models according to your prefrence from")

params = dict(
    user_id="clarifai",
    app_id="ml",
    model_name="llama2-7b-alternative-4k",
    model_url=(
        "https://clarifai.com/clarifai/ml/models/llama2-7b-alternative-4k"
    ),
)

"""
Initialize the LLM
"""
logger.info("Initialize the LLM")

llm_model = Clarifai(model_url=params["model_url"])

llm_model = Clarifai(
    model_name=params["model_name"],
    app_id=params["app_id"],
    user_id=params["user_id"],
)

"""
Call `complete` function
"""
logger.info("Call `complete` function")

llm_reponse = llm_model.complete(
    prompt="write a 10 line rhyming poem about science"
)

logger.debug(llm_reponse)

"""
Call `chat` function
"""
logger.info("Call `chat` function")


messages = [
    ChatMessage(role="user", content="write about climate change in 50 lines")
]
Response = llm_model.chat(messages)

logger.debug(Response)

"""
### Using Inference parameters

Alternatively you can call models with inference parameters.
"""
logger.info("### Using Inference parameters")

inference_params = dict(temperature=str(0.3), max_tokens=20)

llm_reponse = llm_model.complete(
    prompt="What is nuclear fission and fusion?",
    inference_params=params,
)

messages = [ChatMessage(role="user", content="Explain about the big bang")]
Response = llm_model.chat(messages, inference_params=params)

logger.info("\n\n[DONE]", bright=True)