import deeplake
import numpy as np
import matplotlib.pyplot as plt
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
from jet.llm.ollama.embeddings import get_embedding_function
from jet.llm.ollama.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.token.token_utils import get_ollama_tokenizer
from jet.transformers.formatters import format_json
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.readers.file.base import SimpleDirectoryReader
import openai
import os
from deeplake.core.vectorstore import VectorStore

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = '<OPENAI_API_KEY>'
logger.log("ACTIVELOOP_TOKEN:",
           os.environ.get("ACTIVELOOP_TOKEN", "None"), colors=["GRAY", "INFO"])

data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
embed_model = OLLAMA_SMALL_EMBED_MODEL
chunk_size: int = OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]
chunk_overlap: int = 40

# Define file and vector store paths
documents = SimpleDirectoryReader(data_dir).load_data()
splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    tokenizer=get_ollama_tokenizer(embed_model).encode
)
all_nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
# source_text = 'paul_graham_essay.txt'
VECTOR_STORE_PATH = os.path.join(GENERATED_DIR, 'pg_essay_deeplake')
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Read the text file
# with open(source_text, 'r') as f:
#     text = f.read()
# text = "\n\n".join([doc.text for doc in documents])
text = [node.text for node in all_nodes]

# source_text = ",".join([doc.text for doc in documents])
metadata = [node.metadata["file_name"] for node in all_nodes]

# Define embedding function using OpenAI's API


# def embedding_function(texts, model="text-embedding-ada-002"):
#     if isinstance(texts, str):
#         texts = [texts]

#     texts = [t.replace("\n", " ") for t in texts]
#     return [data.embedding for data in openai.embeddings.create(input=texts, model=model).data]
def ollama_embedding_function(texts, model=embed_model):
    if isinstance(texts, str):
        texts = [texts]

    embed_model = OllamaEmbedding(model_name=model)
    results = embed_model.get_general_text_embedding(texts)
    return results


EMBEDDING_FUNCTION = ollama_embedding_function


def create_vector_store(
    store_path: str,
    embedding_function=EMBEDDING_FUNCTION,
    overwrite: bool = False,
    **kwargs
) -> VectorStore:
    # Create a VectorStore instance
    vector_store = VectorStore(
        path=store_path,
        embedding_function=embedding_function,
        overwrite=overwrite,
        **kwargs
    )

    # Add text chunks to the vector store with embeddings
    vector_store.add(
        text=text,
        embedding_function=embedding_function,
        embedding_data=text,
        metadata=metadata,
    )

    # Print vector store summary
    print("Vector store created. Summary:")
    print(vector_store.summary())
    return vector_store


def perform_search(
    prompt: str,
    store_path: str,
    top_k=4,
    embedding_function=EMBEDDING_FUNCTION,
):
    # Perform a vector search
    vector_store = VectorStore(path=store_path)
    results = vector_store.search(
        embedding_data=prompt,
        embedding_function=embedding_function,
        k=top_k,
    )
    return results


# Add this function to visualize the embeddings
# def visualize_vector_store(store_path):
    # ds = deeplake.load(store_path)
    # ds.visualize()  # For jupyter notebook only


def summarize_vector_store(store_path):
    ds = deeplake.load(store_path)
    return ds.summary()


# Visualize the dataset in the main block
if __name__ == "__main__":
    overwrite = False
    prompt = "Tell me about yourself."
    top_k = 4

    # Create the vector store (run once to populate the store)
    logger.debug("Creating vector store...")
    create_vector_store(VECTOR_STORE_PATH, overwrite=overwrite, verbose=True)

    # Visualize the embeddings
    logger.debug("Summarizing dataset...")
    summary = summarize_vector_store(VECTOR_STORE_PATH)
    logger.newline()
    logger.info("Dataset summary:")
    logger.success(summary)

    # Perform a vector search with a sample prompt
    logger.debug("Searching...")
    results = perform_search(prompt, VECTOR_STORE_PATH, top_k)

    # Print top search result
    logger.newline()
    logger.info(f"Top search results ({len(results)}) for query:")
    logger.debug(prompt)
    logger.success(format_json(results))
