from jet.logger import logger
from langchain_core.documents import Document
from langchain_google_community import GoogleTranslateTransformer
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
# Google Translate

[Google Translate](https://translate.google.com/) is a multilingual neural machine translation service developed by Google to translate text, documents and websites from one language into another.

The `GoogleTranslateTransformer` allows you to translate text and HTML with the [Google Cloud Translation API](https://cloud.google.com/translate).

To use it, you should have the `google-cloud-translate` python package installed, and a Google Cloud project with the [Translation API enabled](https://cloud.google.com/translate/docs/setup). This transformer uses the [Advanced edition (v3)](https://cloud.google.com/translate/docs/intro-to-v3).

- [Google Neural Machine Translation](https://en.wikipedia.org/wiki/Google_Neural_Machine_Translation)
- [A Neural Network for Machine Translation, at Production Scale](https://blog.research.google/2016/09/a-neural-network-for-machine.html)
"""
logger.info("# Google Translate")

# %pip install --upgrade --quiet  google-cloud-translate


"""
## Input

This is the document we'll translate
"""
logger.info("## Input")

sample_text = """[Generated with Google Bard]
Subject: Key Business Process Updates

Date: Friday, 27 October 2023

Dear team,

I am writing to provide an update on some of our key business processes.

Sales process

We have recently implemented a new sales process that is designed to help us close more deals and grow our revenue. The new process includes a more rigorous qualification process, a more streamlined proposal process, and a more effective customer relationship management (CRM) system.

Marketing process

We have also revamped our marketing process to focus on creating more targeted and engaging content. We are also using more social media and paid advertising to reach a wider audience.

Customer service process

We have also made some improvements to our customer service process. We have implemented a new customer support system that makes it easier for customers to get help with their problems. We have also hired more customer support representatives to reduce wait times.

Overall, we are very pleased with the progress we have made on improving our key business processes. We believe that these changes will help us to achieve our goals of growing our business and providing our customers with the best possible experience.

If you have any questions or feedback about any of these changes, please feel free to contact me directly.

Thank you,

Lewis Cymbal
CEO, Cymbal Bank
"""

"""
When initializing the `GoogleTranslateTransformer`, you can include the following parameters to configure the requests.

- `project_id`: Google Cloud Project ID.
- `location`: (Optional) Translate model location.
  - Default: `global` 
- `model_id`: (Optional) Translate [model ID][models] to use.
- `glossary_id`: (Optional) Translate [glossary ID][glossaries] to use.
- `api_endpoint`: (Optional) [Regional endpoint][endpoints] to use.

[models]: https://cloud.google.com/translate/docs/advanced/translating-text-v3#comparing-models
[glossaries]: https://cloud.google.com/translate/docs/advanced/glossary
[endpoints]: https://cloud.google.com/translate/docs/advanced/endpoints
"""
logger.info("When initializing the `GoogleTranslateTransformer`, you can include the following parameters to configure the requests.")

documents = [Document(page_content=sample_text)]
translator = GoogleTranslateTransformer(project_id="<YOUR_PROJECT_ID>")

"""
## Output

After translating a document, the result will be returned as a new document with the `page_content` translated into the target language.

You can provide the following keyword parameters to the `transform_documents()` method:

- `target_language_code`: [ISO 639][iso-639] language code of the output document.
    - For supported languages, refer to [Language support][supported-languages].
- `source_language_code`: (Optional) [ISO 639][iso-639] language code of the input document.
    - If not provided, language will be auto-detected.
- `mime_type`: (Optional) [Media Type][media-type] of the input text.
    - Options: `text/plain` (Default), `text/html`.

[iso-639]: https://en.wikipedia.org/wiki/ISO_639
[supported-languages]: https://cloud.google.com/translate/docs/languages
[media-type]: https://en.wikipedia.org/wiki/Media_type
"""
logger.info("## Output")

translated_documents = translator.transform_documents(
    documents, target_language_code="es"
)

for doc in translated_documents:
    logger.debug(doc.metadata)
    logger.debug(doc.page_content)

logger.info("\n\n[DONE]", bright=True)