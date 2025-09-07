from IPython.display import display, Audio, Markdown
from dotenv import load_dotenv
from jet.logger import CustomLogger
from langgraph.graph import StateGraph, END
from ollama import Client
from typing import TypedDict
import io
import os
import re
import shutil
import tempfile


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Building an Intelligent Text-to-Speech Agent with LangGraph and Ollama

## Overview
This tutorial guides you through the process of creating an advanced text-to-speech (TTS) agent using LangGraph and Ollama's APIs. The agent can classify input text, process it based on its content type, and generate corresponding speech output.

## Motivation
In the era of AI and natural language processing, there's a growing need for systems that can intelligently process and vocalize text. This project aims to create a versatile TTS agent that goes beyond simple text-to-speech conversion by understanding and adapting to different types of content.

## Key Components
1. **Content Classification**: Utilizes Ollama's GPT models to categorize input text.
2. **Content Processing**: Applies specific processing based on the content type (general, poem, news, or joke).
3. **Text-to-Speech Conversion**: Leverages Ollama's TTS API to generate audio from processed text.
4. **LangGraph Workflow**: Orchestrates the entire process using a state graph.

## Method
The TTS agent operates through the following high-level steps:

1. **Text Input**: The system receives a text input from the user.
2. **Content Classification**: The input is classified into one of four categories: general, poem, news, or joke.
3. **Content-Specific Processing**: Based on the classification, the text undergoes specific processing:
   - General text remains unchanged
   - Poems are rewritten for enhanced poetic quality
   - News is reformatted into a formal news anchor style
   - Jokes are refined for humor
4. **Text-to-Speech Conversion**: The processed text is converted to speech using an appropriate voice for its content type.
5. **Audio Output**: The generated audio is either saved to a file or played directly, depending on user preferences.

The entire workflow is managed by a LangGraph state machine, ensuring smooth transitions between different processing stages and maintaining context throughout the operation.

## Conclusion
This intelligent TTS agent demonstrates the power of combining language models for content understanding with speech synthesis technology. It offers a more nuanced and context-aware approach to text-to-speech conversion, opening up possibilities for more natural and engaging audio content generation across various applications, from content creation to accessibility solutions.

By leveraging the strengths of GPT models for text processing and Ollama's TTS capabilities, this project showcases how advanced AI technologies can be integrated to create sophisticated, multi-step language processing pipelines.

<div style="text-align: center;">

<img src="../images/tts_poem_generator_agent_langgraph.svg" alt="tts poem generator agent langgraph" style="width:80%; height:auto;">
</div>

## Import necessary libraries and set up environment
"""
logger.info("# Building an Intelligent Text-to-Speech Agent with LangGraph and Ollama")


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
## Initialize Ollama client and define state
"""
logger.info("## Initialize Ollama client and define state")

client = Client()

class AgentState(TypedDict):
    input_text: str
    processed_text: str
    audio_data: bytes
    audio_path: str
    content_type: str

"""
## Define Node Functions
"""
logger.info("## Define Node Functions")

def classify_content(state: AgentState) -> AgentState:
    """Classify the input text into one of four categories: general, poem, news, or joke."""
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Classify the content as one of: 'general', 'poem', 'news', 'joke'."},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["content_type"] = response.message.content.strip().lower()
    return state

def process_general(state: AgentState) -> AgentState:
    """Process general content (no specific processing, return as-is)."""
    state["processed_text"] = state["input_text"]
    return state

def process_poem(state: AgentState) -> AgentState:
    """Process the input text as a poem, rewriting it in a poetic style."""
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Rewrite the following text as a short, beautiful poem:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.message.content.strip()
    return state

def process_news(state: AgentState) -> AgentState:
    """Process the input text as news, rewriting it in a formal news anchor style."""
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Rewrite the following text in a formal news anchor style:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.message.content.strip()
    return state

def process_joke(state: AgentState) -> AgentState:
    """Process the input text as a joke, turning it into a short, funny joke."""
    response = client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Turn the following text into a short, funny joke:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.message.content.strip()
    return state



def text_to_speech(state: AgentState, save_file: bool = False) -> AgentState:
    """
    Converts processed text into speech using a voice mapped to the content type.
    Optionally saves the audio to a file.

    Args:
        state (AgentState): Dictionary containing the processed text and content type.
        save_file (bool, optional): If True, saves the audio to a file. Defaults to False.

    Returns:
        AgentState: Updated state with audio data and file path (if saved).
    """

    voice_map = {
        "general": "alloy",
        "poem": "nova",
        "news": "onyx",
        "joke": "shimmer"
    }
    voice = voice_map.get(state["content_type"], "alloy")

    audio_data = io.BytesIO()

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=state["processed_text"]
    ) as response:
        for chunk in response.iter_bytes():
            audio_data.write(chunk)

    state["audio_data"] = audio_data.getvalue()

    if save_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(state["audio_data"])
            state["audio_path"] = temp_audio.name
    else:
        state["audio_path"] = ""

    return state

"""
## Define and Compile the Graph
"""
logger.info("## Define and Compile the Graph")

workflow = StateGraph(AgentState)

workflow.add_node("classify_content", classify_content)
workflow.add_node("process_general", process_general)
workflow.add_node("process_poem", process_poem)
workflow.add_node("process_news", process_news)
workflow.add_node("process_joke", process_joke)
workflow.add_node("text_to_speech", text_to_speech)

workflow.set_entry_point("classify_content")

workflow.add_conditional_edges(
    "classify_content",
    lambda x: x["content_type"],
    {
        "general": "process_general",
        "poem": "process_poem",
        "news": "process_news",
        "joke": "process_joke",
    }
)

workflow.add_edge("process_general", "text_to_speech")
workflow.add_edge("process_poem", "text_to_speech")
workflow.add_edge("process_news", "text_to_speech")
workflow.add_edge("process_joke", "text_to_speech")

app = workflow.compile()

"""
## A function to convert text to a valid informative filename
"""
logger.info("## A function to convert text to a valid informative filename")

def sanitize_filename(text, max_length=20):
    """Convert text to a valid and concise filename."""
    sanitized = re.sub(r'[^\w\s-]', '', text.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized[:max_length]

"""
## Define Function to Run Agent and Play Audio
"""
logger.info("## Define Function to Run Agent and Play Audio")

def run_tts_agent_and_play(input_text: str, content_type: str, save_file: bool = True):
    result = app.invoke({
        "input_text": input_text,
        "processed_text": "",
        "audio_data": b"",
        "audio_path": "",
        "content_type": content_type
    })

    logger.debug(f"Detected content type: {result['content_type']}")
    logger.debug(f"Processed text: {result['processed_text']}")

    display(Audio(result['audio_data'], autoplay=True))

    if save_file:
        audio_dir = os.path.join('..', 'audio')
        os.makedirs(audio_dir, exist_ok=True)

        sanitized_text = sanitize_filename(input_text)
        file_name = f"{content_type}_{sanitized_text}.mp3"
        file_path = os.path.join(audio_dir, file_name)

        with open(file_path, "wb") as f:
            f.write(result['audio_data'])

        logger.debug(f"Audio saved to: {file_path}")

        github_relative_path = f"../audio/{file_name}"
        display(Markdown(f"[Download {content_type} audio: {sanitized_text}]({github_relative_path})"))

        logger.debug("Note: Audio playback is not supported directly on GitHub. Use the download link to listen to the audio.")
    else:
        logger.debug("Audio not saved to file.")

    return result

"""
## Test the Text-to-Speech Agent
"""
logger.info("## Test the Text-to-Speech Agent")

examples = {
    "general": "The quick brown fox jumps over the lazy dog.",
    "poem": "Roses are red, violets are blue, AI is amazing, and so are you!",
    "news": "Breaking news: Scientists discover a new species of deep-sea creature in the Mariana Trench.",
    "joke": "Why don't scientists trust atoms? Because they make up everything!"
}

for content_type, text in examples.items():
    logger.debug(f"\nProcessing example for {content_type} content:")
    logger.debug(f"Input text: {text}")

    result = run_tts_agent_and_play(text, content_type, save_file=True)

    logger.debug("-" * 50)

logger.debug("All examples processed. You can download the audio files using the links above.")

logger.info("\n\n[DONE]", bright=True)