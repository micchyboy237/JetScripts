from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from jet.logger import CustomLogger
from pprint import pprint
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🇮🇹🇬🇧 Multilingual RAG from a 🎧 podcast


*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*

This notebook shows how to create a multilingual Retrieval Augmented Generation application application, starting from a podcast.

🧰 **Stack**:
- Haystack LLM framework
- OllamaFunctionCalling Whisper model for audio transcription
- Qdrant vector database
- multilingual embedding model: multilingual-e5-large
- multilingual LLM: Mistral Small

## Installation
"""
logger.info("# 🇮🇹🇬🇧 Multilingual RAG from a 🎧 podcast")

# %%capture
# ! pip install -U mistral-haystack haystack-ai qdrant-haystack "openai-whisper>=20231106" pytube "sentence-transformers>=3.0.0" "huggingface_hub>=0.23.0"

"""
## Podcast transcription

- download the audio from Youtube using `pytube`
- transcribe it locally using Haystack's [`LocalWhisperTranscriber`](https://docs.haystack.deepset.ai/docs/localwhispertranscriber) with the `whisper-small` model. We could use bigger models, which take longer to transcribe. We could also call the paid OllamaFunctionCalling API, using [`RemoteWhisperTranscriber`](https://docs.haystack.deepset.ai/docs/remotewhispertranscriber).

Since the transcription takes some time (about 10 minutes), I commented out the following code and will provide the transcription.
"""
logger.info("## Podcast transcription")




"""
## Indexing pipeline

Create an Indexing pipeline that stores chunks of the transcript in the [Qdrant vector database](https://haystack.deepset.ai/integrations/qdrant-document-store).

- [`TextFileToDocument`](https://docs.haystack.deepset.ai/docs/textfiletodocument) converts the transcript into a [Haystack Document](https://docs.haystack.deepset.ai/docs/data-classes#document).
- [`DocumentSplitter`](https://docs.haystack.deepset.ai/docs/documentsplitter) divides the original Document into smaller chunks.
- [`SentenceTransformersDocumentEmbedder`](https://docs.haystack.deepset.ai/docs/sentencetransformersdocumentembedder) computes embeddings(=vector representations) of Documents using a multilingual model, to allow semantic retrieval
- [`DocumentWriter`](https://docs.haystack.deepset.ai/docs/documentwriter) stores the Documents in Qdrant
"""
logger.info("## Indexing pipeline")

# !wget "https://raw.githubusercontent.com/deepset-ai/haystack-cookbook/main/data/multilingual_rag_podcast/podcast_transcript_whisper_small.txt"

# !head --bytes 300 podcast_transcript_whisper_small.txt


document_store = QdrantDocumentStore(
    ":memory:",
    embedding_dim=1024,  # the embedding_dim should match that of the embedding model
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("text_file_converter", TextFileToDocument())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))

indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        model="intfloat/multilingual-e5-large",  # good multilingual model: https://huggingface.co/intfloat/multilingual-e5-large
        device=ComponentDevice.from_str("cuda:0"),    # load the model on GPU
        prefix="passage:",  # as explained in the model card (https://huggingface.co/intfloat/multilingual-e5-large#faq), documents should be prefixed with "passage:"
    ))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("text_file_converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")



res = indexing_pipeline.run({"text_file_converter":{"sources":["podcast_transcript_whisper_small.txt"]}})

document_store.count_documents()

"""
## RAG pipeline

Finally our RAG pipeline: from an Italian podcast 🇮🇹🎧 to answering questions in English 🇬🇧

- [`SentenceTransformersTextEmbedder`](https://docs.haystack.deepset.ai/docs/sentencetransformerstextembedder) transforms the query into a vector that captures its semantics, to allow vector retrieval
- `QdrantRetriever` compares the query and Document embeddings and fetches the Documents most relevant to the query.
- [`ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder) prepares the prompt for the LLM: renders a prompt template and fill in variable values.
- [`MistralChatGenerator`](https://docs.haystack.deepset.ai/docs/mistralchatgenerator) allows using Mistral LLMs. Read their [Quickstart](https://docs.mistral.ai/getting-started/quickstart/) to get an API key.
"""
logger.info("## RAG pipeline")

# from getpass import getpass


# os.environ["MISTRAL_API_KEY"] = getpass("Enter your Mistral API key")

generator = MistralChatGenerator(model="mistral-small-latest")

generator.run([ChatMessage.from_user("Please explain in a fun way why vim is the ultimate IDE")])

template = [ChatMessage.from_user("""
Using only the information contained in these documents in Italian, answer the question using English.
If the answer cannot be inferred from the documents, respond \"I don't know\".
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
""")]

query_pipeline = Pipeline()

query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(
        model="intfloat/multilingual-e5-large",  # good multilingual model: https://huggingface.co/intfloat/multilingual-e5-large
        device=ComponentDevice.from_str("cuda:0"),  # load the model on GPU
        prefix="query:",  # as explained in the model card (https://huggingface.co/intfloat/multilingual-e5-large#faq), queries should be prefixed with "query:"
    ))
query_pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
query_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
query_pipeline.add_component("generator", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "generator")



question = "What is Pointer Podcast?"
results = query_pipeline.run(
    {   "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)

for d in results['generator']['replies']:
  plogger.debug(d.text)

"""
✨ Nice!
"""

def ask_rag(question: str):
  results = query_pipeline.run(
      {
          "text_embedder": {"text": question},
          "prompt_builder": {"question": question},
      }
  )

  for d in results["generator"]["replies"]:
      plogger.debug(d.text)

"""
## Try our multilingual RAG application!
"""
logger.info("## Try our multilingual RAG application!")


questions="""What are some interesting directions in Large Language Models?
What is Haystack?
What is Ollama?
How did Stefano end up working at deepset?
Will open source models achieve the quality of closed ones?
What are the main features of Haystack?
Summarize in a bulleted list the main stages of training a Large Language Model
What is Zephyr?
What is it and why is the quantization of Large Language Models interesting?
Could you point out the names of the hosts and guests of the podcast?""".split("\n")

q = random.choice(questions)
logger.debug(q)
ask_rag(q)

logger.info("\n\n[DONE]", bright=True)