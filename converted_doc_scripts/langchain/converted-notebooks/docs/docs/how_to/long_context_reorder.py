from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_transformers import LongContextReorder
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
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
# How to reorder retrieved results to mitigate the "lost in the middle" effect

Substantial performance degradations in [RAG](/docs/tutorials/rag) applications have been [documented](https://arxiv.org/abs/2307.03172) as the number of retrieved documents grows (e.g., beyond ten). In brief: models are liable to miss relevant information in the middle of long contexts.

By contrast, queries against vector stores will typically return documents in descending order of relevance (e.g., as measured by cosine similarity of [embeddings](/docs/concepts/embedding_models)).

To mitigate the ["lost in the middle"](https://arxiv.org/abs/2307.03172) effect, you can re-order documents after retrieval such that the most relevant documents are positioned at extrema (e.g., the first and last pieces of context), and the least relevant documents are positioned in the middle. In some cases this can help surface the most relevant information to LLMs.

The [LongContextReorder](https://python.langchain.com/api_reference/community/document_transformers/langchain_community.document_transformers.long_context_reorder.LongContextReorder.html) document transformer implements this re-ordering procedure. Below we demonstrate an example.
"""
logger.info("# How to reorder retrieved results to mitigate the "lost in the middle" effect")

# %pip install -qU langchain langchain-community langchain-ollama

"""
First we embed some artificial documents and index them in a basic in-memory vector store. We will use [Ollama](/docs/integrations/providers/ollama/) embeddings, but any LangChain vector store or embeddings model will suffice.
"""
logger.info(
    "First we embed some artificial documents and index them in a basic in-memory vector store. We will use [Ollama](/docs/integrations/providers/ollama/) embeddings, but any LangChain vector store or embeddings model will suffice.")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]

retriever = InMemoryVectorStore.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
query = "What can you tell me about the Celtics?"

docs = retriever.invoke(query)
for doc in docs:
    logger.debug(f"- {doc.page_content}")

"""
Note that documents are returned in descending order of relevance to the query. The `LongContextReorder` document transformer will implement the re-ordering described above:
"""
logger.info("Note that documents are returned in descending order of relevance to the query. The `LongContextReorder` document transformer will implement the re-ordering described above:")


reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

for doc in reordered_docs:
    logger.debug(f"- {doc.page_content}")

"""
Below, we show how to incorporate the re-ordered documents into a simple question-answering chain:
"""
logger.info(
    "Below, we show how to incorporate the re-ordered documents into a simple question-answering chain:")


llm = ChatOllama(model="llama3.2")

prompt_template = """
Given these texts:
-----
{context}
-----
Please answer the following question:
{query}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"],
)

chain = create_stuff_documents_chain(llm, prompt)
response = chain.invoke({"context": reordered_docs, "query": query})
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
