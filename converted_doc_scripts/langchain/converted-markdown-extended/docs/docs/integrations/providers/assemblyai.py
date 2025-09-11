from jet.logger import logger
from langchain_community.document_loaders import AssemblyAIAudioLoaderById
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
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
# AssemblyAI

>[AssemblyAI](https://www.assemblyai.com/) builds `Speech AI` models for tasks like
speech-to-text, speaker diarization, speech summarization, and more.
> `AssemblyAI’s` Speech AI models include accurate speech-to-text for voice data
> (such as calls, virtual meetings, and podcasts), speaker detection, sentiment analysis,
> chapter detection, PII redaction.



## Installation and Setup

Get your [API key](https://www.assemblyai.com/dashboard/signup).

Install the `assemblyai` package.
"""
logger.info("# AssemblyAI")

pip install -U assemblyai

"""
## Document Loader

###  AssemblyAI Audio Transcript

The `AssemblyAIAudioTranscriptLoader` transcribes audio files with the `AssemblyAI API`
and loads the transcribed text into documents.

See a [usage example](/docs/integrations/document_loaders/assemblyai).
"""
logger.info("## Document Loader")


"""
###  AssemblyAI Audio Loader By Id

The `AssemblyAIAudioLoaderById` uses the AssemblyAI API to get an existing
transcription and loads the transcribed text into one or more Documents,
depending on the specified format.
"""
logger.info("###  AssemblyAI Audio Loader By Id")


logger.info("\n\n[DONE]", bright=True)