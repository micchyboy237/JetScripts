from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
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
# ElevenLabs Text2Speech

This notebook shows how to interact with the `ElevenLabs API` to achieve text-to-speech capabilities.

First, you need to set up an ElevenLabs account. You can follow the instructions [here](https://docs.elevenlabs.io/welcome/introduction).
"""
logger.info("# ElevenLabs Text2Speech")

# %pip install --upgrade --quiet  elevenlabs langchain-community


os.environ["ELEVENLABS_API_KEY"] = ""

"""
## Usage
"""
logger.info("## Usage")


text_to_speak = "Hello world! I am the real slim shady"

tts = ElevenLabsText2SpeechTool()
tts.name

"""
We can generate audio, save it to the temporary file and then play it.
"""
logger.info("We can generate audio, save it to the temporary file and then play it.")

speech_file = tts.run(text_to_speak)
tts.play(speech_file)

"""
Or stream audio directly.
"""
logger.info("Or stream audio directly.")

tts.stream_speech(text_to_speak)

"""
## Use within an Agent
"""
logger.info("## Use within an Agent")


llm = Ollama(temperature=0)
tools = load_tools(["eleven_labs_text2speech"])
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

audio_file = agent.run("Tell me a joke and read it out for me.")

tts.play(audio_file)

logger.info("\n\n[DONE]", bright=True)