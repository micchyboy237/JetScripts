import json
from llama_index.core import SimpleDirectoryReader, Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import WebBaseLoader
from getpass import getpass
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


"""
# Migrating from ConversationalRetrievalChain

The [`ConversationalRetrievalChain`](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html) was an all-in one way that combined retrieval-augmented generation with chat history, allowing you to "chat with" your documents.

Advantages of switching to the LCEL implementation are similar to the [`RetrievalQA` migration guide](./retrieval_qa.ipynb):

- Clearer internals. The `ConversationalRetrievalChain` chain hides an entire question rephrasing step which dereferences the initial query against the chat history.
  - This means the class contains two sets of configurable prompts, LLMs, etc.
- More easily return source documents.
- Support for runnable methods like streaming and async operations.

Here are equivalent implementations with custom prompts.
We'll use the following ingestion code to load a [blog post by Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/) on autonomous agents into a local vector store:

## Shared setup

For both versions, we'll need to load the data with the `WebBaseLoader` document loader, split it with `RecursiveCharacterTextSplitter`, and add it to an in-memory `FAISS` vector store.

We will also instantiate a chat model to use.
"""

# %pip install --upgrade --quiet langchain-community langchain langchain-openai faiss-cpu beautifulsoup4

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

# Custom settings
input_dir = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume"
chunk_size = 512
chunk_overlap = 50


# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()
class EnhancedDocument:
    def __init__(self, document: Document):
        self.id = document.node_id
        self.metadata = document.metadata
        self.page_content = document.text


data = SimpleDirectoryReader(input_dir, required_exts=[".md"]).load_data()
data = [EnhancedDocument(d) for d in data]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
all_splits = text_splitter.split_documents(data)

vectorstore = FAISS.from_documents(
    documents=all_splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))

llm = ChatOllama(model="llama3.1")

"""
</details>

## LCEL

<details open>
"""

condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, vectorstore.as_retriever(), condense_question_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

logger.newline()
logger.debug("Streaming response...")
stream_iterator = convo_qa_chain.stream(
    {
        "input": "What are your primary skills?",
        "chat_history": [],
    }
)

query = ""
context = ""
response = ""
for output in stream_iterator:
    if "input" in output:
        query = output['input']
    elif "context" in output:
        context = make_serializable(output['context'])
    elif "answer" in output:
        if not response:
            logger.debug(json.dumps(context, indent=2))
            logger.info(query)
        logger.success(output['answer'], flush=True)
        response += output['answer']
    else:
        logger.debug(output)

"""
</details>

## Next steps

You've now seen how to migrate existing usage of some legacy chains to LCEL.

Next, check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.
"""


logger.info("\n\n[DONE]", bright=True)
