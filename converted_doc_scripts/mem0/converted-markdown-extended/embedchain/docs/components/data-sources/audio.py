from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: "🎤 Audio"
---


To use an audio as data source, just add `data_type` as `audio` and pass in the path of the audio (local or hosted).

We use [Deepgram](https://developers.deepgram.com/docs/introduction) to transcribe the audiot to text, and then use the generated text as the data source.

You would require an Deepgram API key which is available [here](https://console.deepgram.com/signup?jump=keys) to use this feature.

### Without customization
"""
logger.info("### Without customization")


os.environ["DEEPGRAM_API_KEY"] = "153xxx"

app = App()
app.add("introduction.wav", data_type="audio")
response = app.query("What is my name and how old am I?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)