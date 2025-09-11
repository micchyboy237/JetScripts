from PIL import Image
from jet.logger import logger
from langchain_community.embeddings import JinaEmbeddings
from numpy import dot
from numpy.linalg import norm
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Jina

You can check the list of available models from [here](https://jina.ai/embeddings/).

## Installation and setup

Install requirements
"""
logger.info("# Jina")

pip install -U langchain-community

"""
Import libraries
"""
logger.info("Import libraries")


"""
## Embed text and queries with Jina embedding models through JinaAI API
"""
logger.info("## Embed text and queries with Jina embedding models through JinaAI API")

text_embeddings = JinaEmbeddings(
    jina_model_name="jina-embeddings-v2-base-en"
)

text = "This is a test document."

query_result = text_embeddings.embed_query(text)

logger.debug(query_result)

doc_result = text_embeddings.embed_documents([text])

logger.debug(doc_result)

"""
## Embed images and queries with Jina CLIP through JinaAI API
"""
logger.info("## Embed images and queries with Jina CLIP through JinaAI API")

multimodal_embeddings = JinaEmbeddings(jina_model_name="jina-clip-v1")

image = "https://avatars.githubusercontent.com/u/126733545?v=4"

description = "Logo of a parrot and a chain on green background"

im = Image.open(requests.get(image, stream=True).raw)
logger.debug("Image:")
display(im)

image_result = multimodal_embeddings.embed_images([image])

logger.debug(image_result)

description_result = multimodal_embeddings.embed_documents([description])

logger.debug(description_result)

cosine_similarity = dot(image_result[0], description_result[0]) / (
    norm(image_result[0]) * norm(description_result[0])
)

logger.debug(cosine_similarity)

logger.info("\n\n[DONE]", bright=True)