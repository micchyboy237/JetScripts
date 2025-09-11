from jet.logger import logger
from langchain_community.embeddings import ClovaEmbeddings
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
# Clova Embeddings
[Clova](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary) offers an embeddings service

This example goes over how to use LangChain to interact with Clova inference for text embedding.
"""
logger.info("# Clova Embeddings")


os.environ["CLOVA_EMB_API_KEY"] = ""
os.environ["CLOVA_EMB_APIGW_API_KEY"] = ""
os.environ["CLOVA_EMB_APP_ID"] = ""


embeddings = ClovaEmbeddings()

query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)

document_text = ["This is a test doc1.", "This is a test doc2."]
document_result = embeddings.embed_documents(document_text)

logger.info("\n\n[DONE]", bright=True)