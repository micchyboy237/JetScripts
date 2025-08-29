from jet.logger import CustomLogger
from llama_index.embeddings.clarifai import ClarifaiEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/clarifai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Embeddings with Clarifai

LlamaIndex has support for Clarifai embeddings models.

You must have a Clarifai account and a Personal Access Token (PAT) key. 
[Check here](https://clarifai.com/settings/security) to get or create a PAT.

Set CLARIFAI_PAT as an environment variable or You can pass PAT as argument to ClarifaiEmbedding class
"""
logger.info("# Embeddings with Clarifai")

# %pip install llama-index-embeddings-clarifai

# !export CLARIFAI_PAT=YOUR_KEY

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index

"""
Models can be referenced either by the full URL or by the model_name, user ID, and app ID combination.
"""
logger.info("Models can be referenced either by the full URL or by the model_name, user ID, and app ID combination.")


embed_model = ClarifaiEmbedding(
    model_url="https://clarifai.com/clarifai/main/models/BAAI-bge-base-en"
)

embed_model = ClarifaiEmbedding(
    model_name="BAAI-bge-base-en",
    user_id="clarifai",
    app_id="main",
    pat=CLARIFAI_PAT,
)

embeddings = embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
Embed list of texts
"""
logger.info("Embed list of texts")

text = "roses are red violets are blue."
text2 = "Make hay while the sun shines."

embeddings = embed_model._get_text_embeddings([text2, text])
logger.debug(len(embeddings))
logger.debug(embeddings[0][:5])
logger.debug(embeddings[1][:5])

logger.info("\n\n[DONE]", bright=True)