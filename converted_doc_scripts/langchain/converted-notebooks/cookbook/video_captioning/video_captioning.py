from jet.logger import logger
from langchain.chains.video_captioning import VideoCaptioningChain
from langchain.chat_models.ollama import ChatOllama
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
# Video Captioning
This notebook shows how to use VideoCaptioningChain, which is implemented using Langchain's ImageCaptionLoader and AssemblyAI to produce .srt files.

This system autogenerates both subtitles and closed captions from a video URL.

## Installing Dependencies
"""
logger.info("# Video Captioning")



"""
## Imports
"""
logger.info("## Imports")

# import getpass


"""
## Setting up API Keys
"""
logger.info("## Setting up API Keys")

# OPENAI_API_KEY = getpass.getpass("Ollama API Key:")

# ASSEMBLYAI_API_KEY = getpass.getpass("AssemblyAI API Key:")

"""
**Required parameters:**

* llm: The language model this chain will use to get suggestions on how to refine the closed-captions
* assemblyai_key: The API key for AssemblyAI, used to generate the subtitles

**Optional Parameters:**

* verbose (Default: True): Sets verbose mode for downstream chain calls
* use_logging (Default: True): Log the chain's processes in run manager
* frame_skip (Default: None): Choose how many video frames to skip during processing. Increasing it results in faster execution, but less accurate results. If None, frame skip is calculated manually based on the framerate Set this to 0 to sample all frames
* image_delta_threshold (Default: 3000000): Set the sensitivity for what the image processor considers a change in scenery in the video, used to delimit closed captions. Higher = less sensitive
* closed_caption_char_limit (Default: 20): Sets the character limit on closed captions
* closed_caption_similarity_threshold (Default: 80): Sets the percentage value to how similar two closed caption models should be in order to be clustered into one longer closed caption
* use_unclustered_video_models (Default: False): If true, closed captions that could not be clustered will be included. May result in spontaneous behaviour from closed captions such as very short lasting captions or fast-changing captions. Enabling this is experimental and not recommended

## Example run
"""
logger.info("## Example run")

chain = VideoCaptioningChain(
#     llm=ChatOllama(model="llama3.2"),
    assemblyai_key=ASSEMBLYAI_API_KEY,
)

srt_content = chain.run(
    video_file_path="https://ia601200.us.archive.org/9/items/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb.mp4"
)

logger.debug(srt_content)

"""
## Writing output to .srt file
"""
logger.info("## Writing output to .srt file")

with open("output.srt", "w") as file:
    file.write(srt_content)

logger.info("\n\n[DONE]", bright=True)