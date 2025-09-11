from jet.logger import logger
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.document_loaders.hugging_face_dataset import (
HuggingFaceDatasetLoader,
)
import os
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
# HuggingFace dataset

>The [Hugging Face Hub](https://huggingface.co/docs/hub/index) is home to over 5,000 [datasets](https://huggingface.co/docs/hub/index#datasets) in more than 100 languages that can be used for a broad range of tasks across NLP, Computer Vision, and Audio. They used for a diverse range of tasks such as translation,
automatic speech recognition, and image classification.


This notebook shows how to load `Hugging Face Hub` datasets to LangChain.
"""
logger.info("# HuggingFace dataset")


dataset_name = "imdb"
page_content_column = "text"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()

data[:15]

"""
### Example 
In this example, we use data from a dataset to answer a question
"""
logger.info("### Example")


dataset_name = "tweet_eval"
page_content_column = "text"
name = "stance_climate"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name)

index = VectorstoreIndexCreator().from_loaders([loader])

query = "What are the most used hashtag?"
result = index.query(query)

result

logger.info("\n\n[DONE]", bright=True)