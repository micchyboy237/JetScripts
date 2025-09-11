from jet.logger import logger
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_google_community import DocAIParser
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
# Google Cloud Document AI

Document AI is a document understanding platform from Google Cloud to transform unstructured data from documents into structured data, making it easier to understand, analyze, and consume.

Learn more:

- [Document AI overview](https://cloud.google.com/document-ai/docs/overview)
- [Document AI videos and labs](https://cloud.google.com/document-ai/docs/videos)
- [Try it!](https://cloud.google.com/document-ai/docs/drag-and-drop)

The module contains a `PDF` parser based on DocAI from Google Cloud.

You need to install two libraries to use this parser:
"""
logger.info("# Google Cloud Document AI")

# %pip install --upgrade --quiet  langchain-google-community[docai]

"""
First, you need to set up a Google Cloud Storage (GCS) bucket and create your own Optical Character Recognition (OCR) processor as described here: https://cloud.google.com/document-ai/docs/create-processor

The `GCS_OUTPUT_PATH` should be a path to a folder on GCS (starting with `gs://`) and a `PROCESSOR_NAME` should look like `projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID` or `projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID/processorVersions/PROCESSOR_VERSION_ID`. You can get it either programmatically or copy from the `Prediction endpoint` section of the `Processor details` tab in the Google Cloud Console.
"""
logger.info("First, you need to set up a Google Cloud Storage (GCS) bucket and create your own Optical Character Recognition (OCR) processor as described here: https://cloud.google.com/document-ai/docs/create-processor")

GCS_OUTPUT_PATH = "gs://BUCKET_NAME/FOLDER_PATH"
PROCESSOR_NAME = "projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID"


"""
Now, create a `DocAIParser`.
"""
logger.info("Now, create a `DocAIParser`.")

parser = DocAIParser(
    location="us", processor_name=PROCESSOR_NAME, gcs_output_path=GCS_OUTPUT_PATH
)

"""
For this example, you can use an Alphabet earnings report that's uploaded to a public GCS bucket.

[2022Q1_alphabet_earnings_release.pdf](https://storage.googleapis.com/cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf)

Pass the document to the `lazy_parse()` method to
"""
logger.info("For this example, you can use an Alphabet earnings report that's uploaded to a public GCS bucket.")

blob = Blob(
    path="gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf"
)

"""
We'll get one document per page, 11 in total:
"""
logger.info("We'll get one document per page, 11 in total:")

docs = list(parser.lazy_parse(blob))
logger.debug(len(docs))

"""
You can run end-to-end parsing of a blob one-by-one. If you have many documents, it might be a better approach to batch them together and maybe even detach parsing from handling the results of parsing.
"""
logger.info("You can run end-to-end parsing of a blob one-by-one. If you have many documents, it might be a better approach to batch them together and maybe even detach parsing from handling the results of parsing.")

operations = parser.docai_parse([blob])
logger.debug([op.operation.name for op in operations])

"""
You can check whether operations are finished:
"""
logger.info("You can check whether operations are finished:")

parser.is_running(operations)

"""
And when they're finished, you can parse the results:
"""
logger.info("And when they're finished, you can parse the results:")

parser.is_running(operations)

results = parser.get_results(operations)
logger.debug(results[0])

"""
And now we can finally generate Documents from parsed results:
"""
logger.info("And now we can finally generate Documents from parsed results:")

docs = list(parser.parse_from_results(results))

logger.debug(len(docs))

logger.info("\n\n[DONE]", bright=True)