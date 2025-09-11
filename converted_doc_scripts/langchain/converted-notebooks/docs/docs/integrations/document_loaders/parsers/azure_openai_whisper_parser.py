from jet.logger import logger
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import AzureOllamaWhisperParser
from langchain_core.documents.base import Blob
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
# Azure Ollama Whisper Parser

>[Azure Ollama Whisper Parser](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/whisper-overview) is a wrapper around the Azure Ollama Whisper API which utilizes machine learning to transcribe audio files to english text. 
>
>The Parser supports `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, and `.webm`.

The current implementation follows LangChain core principles and can be used with other loaders to handle both audio downloading and parsing. As a result of this the parser will `yield` an `Iterator[Document]`.

## Prerequisites

The service requires Azure credentials, Azure endpoint and Whisper Model deployment, which can be set up by following the guide [here](https://learn.microsoft.com/en-us/azure/ai-services/ollama/whisper-quickstart?tabs=command-line%2Cpython-new%2Cjavascript&pivots=programming-language-python). Furthermore, the required dependencies must be installed.
"""
logger.info("# Azure Ollama Whisper Parser")

# %pip install -Uq  langchain langchain-community ollama

"""
## Example 1

The `AzureOllamaWhisperParser`'s method, `.lazy_parse`, accepts a `Blob` object as a parameter containing the file path of the file to be transcribed.
"""
logger.info("## Example 1")


audio_path = "path/to/your/audio/file"
audio_blob = Blob(path=audio_path)


endpoint = "<your_endpoint>"
key = "<your_api_key"
version = "<your_api_version>"
name = "<your_deployment_name>"

parser = AzureOllamaWhisperParser(
    api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
)

documents = parser.lazy_parse(blob=audio_blob)

for doc in documents:
    logger.debug(doc.page_content)

"""
## Example 2

The `AzureOllamaWhisperParser` can also be used in conjunction with audio loaders, like the `YoutubeAudioLoader` with a `GenericLoader`.
"""
logger.info("## Example 2")


url = ["www.youtube.url.com"]

save_dir = "save/directory/"

name = "<your_deployment_name>"

loader = GenericLoader(
    YoutubeAudioLoader(url, save_dir), AzureOllamaWhisperParser(deployment_name=name)
)

docs = loader.load()

for doc in documents:
    logger.debug(doc.page_content)

logger.info("\n\n[DONE]", bright=True)