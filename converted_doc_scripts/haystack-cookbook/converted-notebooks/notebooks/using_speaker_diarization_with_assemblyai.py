from assemblyai_haystack.transcriber import AssemblyAITranscriber
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Speaker Diarization with AssemblyAI

> 📚 This cookbook has an accompanying article with a complete walkthrough "[Level up Your RAG Application with Speaker Diarization](https://haystack.deepset.ai/blog/level-up-rag-with-speaker-diarization)"

LLMs excel with text data, answering complex questions without manual reading or searching. When dealing with audio or video, providing transcription is key. Transcription captures spoken content of the audio or video, but in multi-speaker recordings, it may miss non-verbal information and fail to convey speaker count or individual remarks. Therefore, to maximize the LLM's potential with such recordings, **Speaker Diarization** is essential!

In this example, we'll build a RAG application with speaker labels for audio files. This application will use [Haystack](https://github.com/deepset-ai/haystack) and speaker diarization models by [AssemblyAI](https://www.assemblyai.com/).

📚 Useful Sources:
* [Integration: AssemblyAI](https://haystack.deepset.ai/integrations/assemblyai)

## Install the Dependencies
"""
logger.info("# Speaker Diarization with AssemblyAI")

# %%bash
pip install haystack
pip install assemblyai-haystack
pip install "sentence-transformers>=3.0.0"
pip install "huggingface_hub>=0.23.0"
pip install --upgrade gdown

"""
## Download The Audio Files

We extracted the audio from youtube videos and saved them in a Google Drive Folder for you: https://drive.google.com/drive/folders/10zsFuHmj3oytYMyGrLdytpW-6JzT9T_W?usp=drive_link

You can run the code below to download the audio files to this colab notebook under "Files" tab on the left bar.
"""
logger.info("## Download The Audio Files")

# !gdown https://drive.google.com/drive/folders/10zsFuHmj3oytYMyGrLdytpW-6JzT9T_W -O "/content" --folder

"""
## Add Your API Keys

Enter the API keys from [AssemblyAI](https://www.assemblyai.com/) and [Hugging Face](https://huggingface.co/settings/tokens):
"""
logger.info("## Add Your API Keys")

# from getpass import getpass

# ASSEMBLYAI_API_KEY = getpass("Enter your ASSEMBLYAI_API_KEY: ")
# os.environ["HF_API_TOKEN"] = getpass("HF_API_TOKEN: ")

"""
## Index Speaker Labels to Your DocumentStore

Build a pipeline to generate speaker labels and index them into a DocumentStore with their embeddings. In this pipeline, you need:

* [InMemoryDocumentStore](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore): to store your documents without external dependencies or extra setup
* [AssemblyAITranscriber](https://github.com/AssemblyAI/assemblyai-haystack/blob/main/assemblyai_haystack/transcriber.py): to create speaker_labels for the given audio file and convert them into Haystack Documents
* [DocumentSplitter](https://docs.haystack.deepset.ai/docs/documentsplitter): to split your documents into smaller chunks
* [SentenceTransformersDocumentEmbedder](https://docs.haystack.deepset.ai/docs/sentencetransformersdocumentembedder): to create embeddings for each document using sentence-transformers models
* [DocumentWriter](https://docs.haystack.deepset.ai/docs/documentwriter): to write these documents into your document store

Note: The speaker information will be saved in the `meta` of the Document object
"""
logger.info("## Index Speaker Labels to Your DocumentStore")


speaker_document_store = InMemoryDocumentStore()
transcriber = AssemblyAITranscriber(api_key=ASSEMBLYAI_API_KEY)
speaker_splitter = DocumentSplitter(
    split_by = "sentence",
    split_length = 10,
    split_overlap = 1
)
speaker_embedder = SentenceTransformersDocumentEmbedder(device=ComponentDevice.from_str("cuda:0"))
speaker_writer = DocumentWriter(speaker_document_store, policy=DuplicatePolicy.SKIP)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=transcriber, name="transcriber")
indexing_pipeline.add_component(instance=speaker_splitter, name="speaker_splitter")
indexing_pipeline.add_component(instance=speaker_embedder, name="speaker_embedder")
indexing_pipeline.add_component(instance=speaker_writer, name="speaker_writer")

indexing_pipeline.connect("transcriber.speaker_labels", "speaker_splitter")
indexing_pipeline.connect("speaker_splitter", "speaker_embedder")
indexing_pipeline.connect("speaker_embedder", "speaker_writer")

"""
Give an `audio_file_path` and run your pipeline
"""
logger.info("Give an `audio_file_path` and run your pipeline")

audio_file_path = "/content/Panel_Discussion.mp3" #@param ["/content/Netflix_Q4_2023_Earnings_Interview.mp3", "/content/Working_From_Home_Debate.mp3", "/content/Panel_Discussion.mp3"]
indexing_pipeline.run(
    {
        "transcriber": {
            "file_path": audio_file_path,
            "summarization": None,
            "speaker_labels": True
        },
    }
)

"""
## RAG Pipeline with Speaker Labels

Build a RAG pipeline to generate answers to questions about the recording. Ensure that speaker information (provided through the metadata of the document) is included in the prompt for the LLM to distinguish who said what. For this pipeline, you need:

* [SentenceTransformersTextEmbedder](https://docs.haystack.deepset.ai/docs/sentencetransformerstextembedder): To create an embedding for the user query using sentence-transformers models
* [InMemoryEmbeddingRetriever](https://docs.haystack.deepset.ai/docs/inmemoryembeddingretriever): to retrieve `top_k` relevant documents to the user query
* [PromptBuilder](https://docs.haystack.deepset.ai/docs/promptbuilder): to provide a RAG prompt template with instructions to be filled with retrieved documents and the user query
* [HuggingFaceAPIGenerator](https://docs.haystack.deepset.ai/docs/huggingfaceapigenerator): to infer models served through Hugging Face free Serverless Inference API or Hugging Face TGI

> The LLM in the example ([`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)) is a gated model. Make sure you have access to the model.
"""
logger.info("## RAG Pipeline with Speaker Labels")


prompt = """
You will be provided with a transcription of a recording with each sentence or group of sentences attributed to a Speaker by the word "Speaker" followed by a letter representing the person uttering that sentence. Answer the given question based on the given context.
If you think that given transcription is not enough to answer the question, say so.

Transcription:
{% for doc in documents %}
  {% if doc.meta["speaker"] %} Speaker {{doc.meta["speaker"]}}: {% endif %}{{doc.content}}
{% endfor %}
Question: {{ question }}
<|end_of_turn|>
Answer:
"""

retriever = InMemoryEmbeddingRetriever(speaker_document_store)
text_embedder = SentenceTransformersTextEmbedder(device=ComponentDevice.from_str("cuda:0"))
answer_generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    generation_kwargs={"max_new_tokens":500})
prompt_builder = PromptBuilder(template=prompt)

speaker_rag_pipe = Pipeline()
speaker_rag_pipe.add_component("text_embedder", text_embedder)
speaker_rag_pipe.add_component("retriever", retriever)
speaker_rag_pipe.add_component("prompt_builder", prompt_builder)
speaker_rag_pipe.add_component("llm", answer_generator)

speaker_rag_pipe.connect("text_embedder.embedding", "retriever.query_embedding")
speaker_rag_pipe.connect("retriever.documents", "prompt_builder.documents")
speaker_rag_pipe.connect("prompt_builder.prompt", "llm.prompt")

"""
## Test RAG with Speaker Labels
"""
logger.info("## Test RAG with Speaker Labels")

question = "What are each speakers' opinions on building in-house or using third parties?" # @param ["What are the two opposing opinions and how many people are on each side?", "What are each speakers' opinions on building in-house or using third parties?", "How many people are speaking in this recording?" ,"How many speakers and moderators are in this call?"]

result = speaker_rag_pipe.run({
    "prompt_builder":{"question": question},
    "text_embedder":{"text": question},
    "retriever":{"top_k": 10}
})
result["llm"]["replies"][0]

logger.info("\n\n[DONE]", bright=True)