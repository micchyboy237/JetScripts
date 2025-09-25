from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
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
# Tutorial: Prepare your own documents for vector search
"""
logger.info("# Tutorial: Prepare your own documents for vector search")

# %pip install langchain pypdf langchain-ollama --quiet

"""
## 1. Upload your documents
First, remove the existing files in the `/docs` folder and add your own PDF files. Then, run the cells below
"""
logger.info("## 1. Upload your documents")


docs_dir = "./docs"

all_files = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir)
             if os.path.isfile(os.path.join(docs_dir, f))]

documents = []
for file_path in all_files:
    try:
        loader = PyPDFLoader(
            file_path=file_path,
        )
        docs = loader.load()
        documents.extend(docs)
        logger.debug(f"Loaded {len(docs)} chunks from {file_path}")
    except Exception as e:
        logger.debug(f"Error loading {file_path}: {e}")

logger.debug(f"Loaded total of {len(documents)} document chunks")

"""
## 2. Chunk documents
Split large documents into smaller chunks for better embedding quality.
"""
logger.info("## 2. Chunk documents")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
logger.debug(f"Created {len(chunks)} chunks")

for chunk in chunks:
    logger.debug(chunk.page_content)
    logger.debug("-"*100)

"""
## 3. Generate embeddings
Use Ollama embeddings to encode each chunk into a vector.
"""
logger.info("## 3. Generate embeddings")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
## 4. Store embeddings in Chroma
Initialize a Chroma vector store and persist it locally.

If you run into a  "OperationalError: attempt to write a readonly database" - restart the kernel and rerun the notebook.
"""
logger.info("## 4. Store embeddings in Chroma")


vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="db",
    collection_name="my_custom_index"
)
vectordb.persist()

"""
## 5. Example similarity search
Perform a similarity search query on your vector store.
"""
logger.info("## 5. Example similarity search")

query = "robotics"
results = vectordb.similarity_search(query, k=5)
for i, doc in enumerate(results):
    logger.debug(f"Result {i+1}: {doc.page_content}...\n")

logger.info("\n\n[DONE]", bright=True)