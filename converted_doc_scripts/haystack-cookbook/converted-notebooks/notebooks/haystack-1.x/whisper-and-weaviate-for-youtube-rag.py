from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from jet.logger import CustomLogger
from pytube import YouTube
from weaviate.embedded import EmbeddedOptions
import os
import shutil
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
📚 Check out the [**Talk to YouTube Videos with Haystack Pipelines**](https://haystack.deepset.ai/blog/talk-to-youtube-videos-with-haystack-pipelines) article for a detailed run through of this example.

## Install the Dependencies
"""
logger.info("## Install the Dependencies")

# !pip install pytube
# !pip install farm-haystack[weaviate,inference,file-conversion,preprocessing]

"""
## (If Needed) Set Your API Token for desired the Model Provider
"""
logger.info("## (If Needed) Set Your API Token for desired the Model Provider")

# from getpass import getpass

# api_key = getpass("Enter OllamaFunctionCallingAdapter API key:")

"""
## The Indexing Pipelne
"""
logger.info("## The Indexing Pipelne")


client = weaviate.Client(
  embedded_options=weaviate.embedded.EmbeddedOptions()
)


document_store = WeaviateDocumentStore(port=6666)


def youtube2audio (url: str):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()


preprocessor = PreProcessor()
embedder = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
whisper = WhisperTranscriber(api_key=api_key)

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=whisper, name="Whisper", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["Whisper"])
indexing_pipeline.add_node(component=embedder, name="Embedder", inputs=["Preprocessor"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Embedder"])

"""
### Run the Indexing Pipeline
"""
logger.info("### Run the Indexing Pipeline")

videos = ["https://www.youtube.com/watch?v=h5id4erwD4s", "https://www.youtube.com/watch?v=iFUeV3aYynI"]

for video in videos:
  file_path = youtube2audio(video)
  indexing_pipeline.run(file_paths=[file_path])

"""
## The RAG Pipeline
"""
logger.info("## The RAG Pipeline")


video_qa_prompt = PromptTemplate(prompt="You will be provided some transcripts from the AI Engineer livestream. Please answer the query based on what is said in the livestream.\n"
                                        "Video Transcripts: {join(documents)}\n"
                                        "Query: {query}\n"
                                        "Answer:", output_parser = AnswerParser())

prompt_node = PromptNode(model_name_or_path="gpt-4", api_key=api_key, default_prompt_template=video_qa_prompt)

video_rag_pipeline = Pipeline()
video_rag_pipeline.add_node(component=embedder, name="Retriever", inputs=["Query"])
video_rag_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

"""
### Run the RAG Pipeline
"""
logger.info("### Run the RAG Pipeline")

result = video_rag_pipeline.run("Why do we do chunking?")
logger.debug(result['answers'][0].answer)

logger.info("\n\n[DONE]", bright=True)