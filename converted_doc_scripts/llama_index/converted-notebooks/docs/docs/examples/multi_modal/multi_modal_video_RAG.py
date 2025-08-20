from PIL import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import MLXMultiModal
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from moviepy.editor import VideoFileClip
from pathlib import Path
from pprint import pprint
from pytube import YouTube
import json
import matplotlib.pyplot as plt
import os
import shutil
import speech_recognition as sr


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/multi_modal_video_RAG.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multimodal RAG for processing videos using MLX GPT4V and LanceDB vectorstore

In this notebook, we showcase a Multimodal RAG architecture designed for video processing. We utilize MLX GPT4V MultiModal LLM class that employs [CLIP](https://github.com/openai/CLIP) to generate multimodal embeddings. Furthermore, we use [LanceDBVectorStore](https://docs.llamaindex.ai/en/latest/examples/vector_stores/LanceDBIndexDemo.html#) for efficient vector storage.



Steps:
1. Download video from YouTube, process and store it.

2. Build Multi-Modal index and vector store for both texts and images.

3. Retrieve relevant images and context, use both to augment the prompt.

4. Using GPT4V for reasoning the correlations between the input query and augmented data and generating final response.
"""
logger.info("# Multimodal RAG for processing videos using MLX GPT4V and LanceDB vectorstore")

# %pip install llama-index-vector-stores-lancedb
# %pip install llama-index-multi-modal-llms-openai

# %pip install llama-index-multi-modal-llms-openai
# %pip install llama-index-vector-stores-lancedb
# %pip install llama-index-embeddings-clip

# %pip install llama_index ftfy regex tqdm
# %pip install -U openai-whisper
# %pip install git+https://github.com/openai/CLIP.git
# %pip install torch torchvision
# %pip install matplotlib scikit-image
# %pip install lancedb
# %pip install moviepy
# %pip install pytube
# %pip install pydub
# %pip install SpeechRecognition
# %pip install ffmpeg-python
# %pip install soundfile



# OPENAI_API_KEY = ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
#### Set configuration for input below
"""
logger.info("#### Set configuration for input below")

video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"

filepath = output_video_path + "input_vid.mp4"
Path(output_folder).mkdir(parents=True, exist_ok=True)

"""
#### Download and process videos into appropriate format for generating/storing embeddings
"""
logger.info("#### Download and process videos into appropriate format for generating/storing embeddings")



def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 7:
                break

def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename="input_vid.mp4"
    )
    return metadata


def video_to_images(video_path, output_folder):
    """
    Convert a video to a sequence of images and save them to the output folder.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.

    """
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(
        os.path.join(output_folder, "frame%04d.png"), fps=0.2
    )


def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)


def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library.

    Parameters:
    audio_path (str): The path to the audio file.

    Returns:
    test (str): The text recognized from the audio.

    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            logger.debug("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            logger.debug(f"Could not request results from service; {e}")

    return text

try:
    metadata_vid = download_video(video_url, output_video_path)
    video_to_images(filepath, output_folder)
    video_to_audio(filepath, output_audio_path)
    text_data = audio_to_text(output_audio_path)

    with open(output_folder + "output_text.txt", "w") as file:
        file.write(text_data)
    logger.debug("Text data saved to file")
    file.close()
    os.remove(output_audio_path)
    logger.debug("Audio file removed")

except Exception as e:
    raise e

"""
#### Create the multi-modal index
"""
logger.info("#### Create the multi-modal index")





text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

documents = SimpleDirectoryReader(output_folder).load_data()

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

"""
#### Use index as retriever to fetch top k (5 in this example) results from the multimodal vector index
"""
logger.info("#### Use index as retriever to fetch top k (5 in this example) results from the multimodal vector index")

retriever_engine = index.as_retriever(
    similarity_top_k=5, image_similarity_top_k=5
)

"""
#### Set the RAG  prompt template
"""
logger.info("#### Set the RAG  prompt template")


metadata_str = json.dumps(metadata_vid)

qa_tmpl_str = (
    "Given the provided information, including relevant images and retrieved context from the video, \
 accurately and precisely answer the query without any additional prior knowledge.\n"
    "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "Metadata for video: {metadata_str} \n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

"""
#### Retrieve most similar text/image embeddings baseed on user query from the DB
"""
logger.info("#### Retrieve most similar text/image embeddings baseed on user query from the DB")



def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

"""
#### Add query now, fetch relevant details including images and augment the prompt template
"""
logger.info("#### Add query now, fetch relevant details including images and augment the prompt template")

query_str = "Using examples from video, explain all things covered in the video regarding the gaussian function"

img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
image_documents = SimpleDirectoryReader(
    input_dir=output_folder, input_files=img
).load_data()
context_str = "".join(txt)
plot_images(img)

"""
#### Generate final response using GPT4V
"""
logger.info("#### Generate final response using GPT4V")


openai_mm_llm = MLXMultiModal(
#     model="qwen3-1.7b-4bit", api_key=OPENAI_API_KEY, max_new_tokens=1500
)


response_1 = openai_mm_llm.complete(
    prompt=qa_tmpl_str.format(
        context_str=context_str, query_str=query_str, metadata_str=metadata_str
    ),
    image_documents=image_documents,
)

plogger.debug(response_1.text)

logger.info("\n\n[DONE]", bright=True)