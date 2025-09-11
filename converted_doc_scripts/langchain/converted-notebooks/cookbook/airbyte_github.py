from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_airbyte import AirbyteLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import os
import shutil
import tiktoken


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

# %pip install -qU langchain-airbyte langchain_chroma

# import getpass

# GITHUB_TOKEN = getpass.getpass()


loader = AirbyteLoader(
    source="source-github",
    stream="pull_requests",
    config={
        "credentials": {"personal_access_token": GITHUB_TOKEN},
        "repositories": ["langchain-ai/langchain"],
    },
    template=PromptTemplate.from_template(
        """# {title}
by {user[login]}

{body}"""
    ),
    include_metadata=False,
)
docs = loader.load()

logger.debug(docs[-2].page_content)

len(docs)


enc = tiktoken.get_encoding("cl100k_base")

vectorstore = Chroma.from_documents(
    docs,
    embedding=OllamaEmbeddings(
        disallowed_special=(enc.special_tokens_set - {"<|endofprompt|>"})
    ),
)

retriever = vectorstore.as_retriever()

retriever.invoke("pull requests related to IBM")

logger.info("\n\n[DONE]", bright=True)
