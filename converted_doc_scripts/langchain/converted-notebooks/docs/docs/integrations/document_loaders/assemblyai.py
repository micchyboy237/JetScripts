from jet.logger import logger
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders.assemblyai import TranscriptFormat
import assemblyai as aai
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
# AssemblyAI Audio Transcripts

The `AssemblyAIAudioTranscriptLoader` allows to transcribe audio files with the [AssemblyAI API](https://www.assemblyai.com) and loads the transcribed text into documents.

To use it, you should have the `assemblyai` python package installed, and the
environment variable `ASSEMBLYAI_API_KEY` set with your API key. Alternatively, the API key can also be passed as an argument.

More info about AssemblyAI:

- [Website](https://www.assemblyai.com/)
- [Get a Free API key](https://www.assemblyai.com/dashboard/signup)
- [AssemblyAI API Docs](https://www.assemblyai.com/docs)

## Installation

First, you need to install the `assemblyai` python package.

You can find more info about it inside the [assemblyai-python-sdk GitHub repo](https://github.com/AssemblyAI/assemblyai-python-sdk).
"""
logger.info("# AssemblyAI Audio Transcripts")

# %pip install --upgrade --quiet  assemblyai

"""
## Example

The `AssemblyAIAudioTranscriptLoader` needs at least the `file_path` argument. Audio files can be specified as an URL or a local file path.
"""
logger.info("## Example")


audio_file = "https://storage.googleapis.com/aai-docs-samples/nbc.mp3"

loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file)

docs = loader.load()

"""
Note: Calling `loader.load()` blocks until the transcription is finished.

The transcribed text is available in the `page_content`:
"""
logger.info("Note: Calling `loader.load()` blocks until the transcription is finished.")

docs[0].page_content

"""
```
"Load time, a new president and new congressional makeup. Same old ..."
```

The `metadata` contains the full JSON response with more meta information:
"""
logger.info("The `metadata` contains the full JSON response with more meta information:")

docs[0].metadata

"""
```
{'language_code': <LanguageCode.en_us: 'en_us'>,
 'audio_url': 'https://storage.googleapis.com/aai-docs-samples/nbc.mp3',
 'punctuate': True,
 'format_text': True,
  ...
}
```

## Transcript Formats

You can specify the `transcript_format` argument for different formats.

Depending on the format, one or more documents are returned. These are the different `TranscriptFormat` options:

- `TEXT`: One document with the transcription text
- `SENTENCES`: Multiple documents, splits the transcription by each sentence
- `PARAGRAPHS`: Multiple documents, splits the transcription by each paragraph
- `SUBTITLES_SRT`: One document with the transcript exported in SRT subtitles format
- `SUBTITLES_VTT`: One document with the transcript exported in VTT subtitles format
"""
logger.info("## Transcript Formats")


loader = AssemblyAIAudioTranscriptLoader(
    file_path="./your_file.mp3",
    transcript_format=TranscriptFormat.SENTENCES,
)

docs = loader.load()

"""
## Transcription Config

You can also specify the `config` argument to use different audio intelligence models.

Visit the [AssemblyAI API Documentation](https://www.assemblyai.com/docs) to get an overview of all available models!
"""
logger.info("## Transcription Config")


config = aai.TranscriptionConfig(
    speaker_labels=True, auto_chapters=True, entity_detection=True
)

loader = AssemblyAIAudioTranscriptLoader(file_path="./your_file.mp3", config=config)

"""
## Pass the API Key as argument

Next to setting the API key as environment variable `ASSEMBLYAI_API_KEY`, it is also possible to pass it as argument.
"""
logger.info("## Pass the API Key as argument")

loader = AssemblyAIAudioTranscriptLoader(
    file_path="./your_file.mp3"
)

logger.info("\n\n[DONE]", bright=True)