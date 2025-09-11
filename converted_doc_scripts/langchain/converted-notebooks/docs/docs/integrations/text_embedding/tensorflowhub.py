from jet.logger import logger
from langchain_community.embeddings import TensorflowHubEmbeddings
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
# TensorFlow Hub

>[TensorFlow Hub](https://www.tensorflow.org/hub) is a repository of trained machine learning models ready for fine-tuning and deployable anywhere. Reuse trained models like `BERT` and `Faster R-CNN` with just a few lines of code.
>
>
Let's load the TensorflowHub Embedding class.
"""
logger.info("# TensorFlow Hub")


embeddings = TensorflowHubEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)

doc_results = embeddings.embed_documents(["foo"])

doc_results

logger.info("\n\n[DONE]", bright=True)