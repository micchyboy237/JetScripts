from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OllamaWhisperParser,
    OllamaWhisperParserLocal,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# YouTube audio

Building chat or QA applications on YouTube videos is a topic of high interest.

Below we show how to easily go from a `YouTube url` to `audio of the video` to `text` to `chat`!

We wil use the `OllamaWhisperParser`, which will use the Ollama Whisper API to transcribe audio to text,
and the  `OllamaWhisperParserLocal` for local support and running on private clouds or on premise.

# Note: You will need to have an `OPENAI_API_KEY` supplied.
"""
logger.info("# YouTube audio")


"""
We will use `yt_dlp` to download audio for YouTube urls.

We will use `pydub` to split downloaded audio files (such that we adhere to Whisper API's 25MB file size limit).
"""
logger.info("We will use `yt_dlp` to download audio for YouTube urls.")

# %pip install --upgrade --quiet  yt_dlp
# %pip install --upgrade --quiet  pydub
# %pip install --upgrade --quiet  librosa

"""
### YouTube url to text

Use `YoutubeAudioLoader` to fetch / download the audio files.

Then, ues `OllamaWhisperParser()` to transcribe them to text.

Let's take the first lecture of Andrej Karpathy's YouTube course as an example!
"""
logger.info("### YouTube url to text")

local = False

urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

save_dir = "~/Downloads/YouTube"

if local:
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir), OllamaWhisperParserLocal()
    )
else:
    loader = GenericLoader(YoutubeAudioLoader(
        urls, save_dir), OllamaWhisperParser())
docs = loader.load()

docs[0].page_content[0:500]

"""
### Building a chat app from YouTube video

Given `Documents`, we can easily enable chat / question+answering.
"""
logger.info("### Building a chat app from YouTube video")


combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = FAISS.from_texts(splits, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

query = "Why do we need to zero out the gradient before backprop at each step?"
qa_chain.run(query)

query = "What is the difference between an encoder and decoder?"
qa_chain.run(query)

query = "For any token, what are x, k, v, and q?"
qa_chain.run(query)

logger.info("\n\n[DONE]", bright=True)
