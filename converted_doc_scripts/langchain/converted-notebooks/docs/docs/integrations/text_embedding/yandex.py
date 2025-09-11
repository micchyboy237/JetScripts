from jet.logger import logger
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
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
# YandexGPT

This notebook goes over how to use Langchain with [YandexGPT](https://cloud.yandex.com/en/services/yandexgpt) embeddings models.

To use, you should have the `yandexcloud` python package installed.
"""
logger.info("# YandexGPT")

# %pip install --upgrade --quiet  yandexcloud

"""
First, you should [create service account](https://cloud.yandex.com/en/docs/iam/operations/sa/create) with the `ai.languageModels.user` role.

Next, you have two authentication options:
- [IAM token](https://cloud.yandex.com/en/docs/iam/operations/iam-token/create-for-sa).
    You can specify the token in a constructor parameter `iam_token` or in an environment variable `YC_IAM_TOKEN`.
- [API key](https://cloud.yandex.com/en/docs/iam/operations/api-key/create)
    You can specify the key in a constructor parameter `api_key` or in an environment variable `YC_API_KEY`.

To specify the model you can use `model_uri` parameter, see [the documentation](https://cloud.yandex.com/en/docs/yandexgpt/concepts/models#yandexgpt-embeddings) for more details.

By default, the latest version of `text-search-query` is used from the folder specified in the parameter `folder_id` or `YC_FOLDER_ID` environment variable.
"""
logger.info("First, you should [create service account](https://cloud.yandex.com/en/docs/iam/operations/sa/create) with the `ai.languageModels.user` role.")


embeddings = YandexGPTEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])

query_result[:5]

doc_result[0][:5]

logger.info("\n\n[DONE]", bright=True)