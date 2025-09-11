from jet.logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import GCSFileLoader
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
# Google Cloud Storage File

>[Google Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.

This covers how to load document objects from an `Google Cloud Storage (GCS) file object (blob)`.
"""
logger.info("# Google Cloud Storage File")

# %pip install --upgrade --quiet  langchain-google-community[gcs]


loader = GCSFileLoader(project_name="aist", bucket="testing-hwc", blob="fake.docx")

loader.load()

"""
If you want to use an alternative loader, you can provide a custom function, for example:
"""
logger.info("If you want to use an alternative loader, you can provide a custom function, for example:")



def load_pdf(file_path):
    return PyPDFLoader(file_path)


loader = GCSFileLoader(
    project_name="aist", bucket="testing-hwc", blob="fake.pdf", loader_func=load_pdf
)

logger.info("\n\n[DONE]", bright=True)