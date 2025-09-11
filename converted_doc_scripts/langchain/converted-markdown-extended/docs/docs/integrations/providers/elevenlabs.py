from jet.logger import logger
from langchain_community.tools import ElevenLabsText2SpeechTool
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
# ElevenLabs

>[ElevenLabs](https://elevenlabs.io/about) is a voice AI research & deployment company
> with a mission to make content universally accessible in any language & voice.
>
>`ElevenLabs` creates the most realistic, versatile and contextually-aware
> AI audio, providing the ability to generate speech in hundreds of
> new and existing voices in 29 languages.

## Installation and Setup

First, you need to set up an ElevenLabs account. You can follow the
[instructions here](https://docs.elevenlabs.io/welcome/introduction).

Install the Python package:
"""
logger.info("# ElevenLabs")

pip install elevenlabs

"""
## Tools

See a [usage example](/docs/integrations/tools/eleven_labs_tts).
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)