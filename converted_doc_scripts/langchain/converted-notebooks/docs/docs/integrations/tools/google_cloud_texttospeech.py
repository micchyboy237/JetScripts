from jet.logger import logger
from langchain_community.tools import GoogleCloudTextToSpeechTool
from langchain_google_community import TextToSpeechTool
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
# Google Cloud Text-to-Speech

>[Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech) enables developers to synthesize natural-sounding speech with 100+ voices, available in multiple languages and variants. It applies DeepMind’s groundbreaking research in WaveNet and Google’s powerful neural networks to deliver the highest fidelity possible.
>
>It supports multiple languages, including English, German, Polish, Spanish, Italian, French, Portuguese, and Hindi.

This notebook shows how to interact with the `Google Cloud Text-to-Speech API` to achieve speech synthesis capabilities.

First, you need to set up an Google Cloud project. You can follow the instructions [here](https://cloud.google.com/text-to-speech/docs/before-you-begin).
"""
logger.info("# Google Cloud Text-to-Speech")

# !pip install --upgrade langchain-google-community[texttospeech]

"""
## Instantiation
"""
logger.info("## Instantiation")


"""
## Deprecated GoogleCloudTextToSpeechTool
"""
logger.info("## Deprecated GoogleCloudTextToSpeechTool")


text_to_speak = "Hello world!"

tts = GoogleCloudTextToSpeechTool()
tts.name

"""
We can generate audio, save it to the temporary file and then play it.
"""
logger.info("We can generate audio, save it to the temporary file and then play it.")

speech_file = tts.run(text_to_speak)

logger.info("\n\n[DONE]", bright=True)