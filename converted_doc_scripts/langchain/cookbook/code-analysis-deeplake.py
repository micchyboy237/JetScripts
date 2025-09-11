import fnmatch
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from jet.adapters.langchain.chat_ollama import ChatOllama

initialize_ollama_settings()

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

DATASET_PATH = f"{GENERATED_DIR}/langchain-code"
os.makedirs(DATASET_PATH, exist_ok=True)


"""
# Use LangChain, GPT and Activeloop's Deep Lake to work with code base
In this tutorial, we are going to use Langchain + Activeloop's Deep Lake with GPT to analyze the code base of the LangChain itself.
"""

"""
## Design
"""

"""
1. Prepare data:
   1. Upload all python project files using the `langchain_community.document_loaders.TextLoader`. We will call these files the **documents**.
   2. Split all documents to chunks using the `langchain_text_splitters.CharacterTextSplitter`.
   3. Embed chunks and upload them into the DeepLake using `langchain.embeddings.openai.OllamaEmbeddings` and `langchain_community.vectorstores.DeepLake`
2. Question-Answering:
   1. Build a chain from `langchain.chat_models.ChatOllama` and `langchain.chains.ConversationalRetrievalChain`
   2. Prepare questions.
   3. Get answers running the chain.
"""

"""
## Implementation
"""

"""
### Integration preparations
"""

"""
We need to set up keys for external services and install necessary python libraries.
"""


"""
Set up Ollama embeddings, Deep Lake multi-modal vector store api and authenticate. 

For full documentation of Deep Lake please follow https://docs.activeloop.ai/ and API reference https://docs.deeplake.ai/en/latest/
"""

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
Authenticate into Deep Lake if you want to create your own dataset and publish it. You can get an API key from the platform at [app.activeloop.ai](https://app.activeloop.ai)
"""

# activeloop_token = getpass("Activeloop Token:")
# os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

"""
### Prepare data
"""

"""
Load all repository files. Here we assume this notebook is downloaded as the part of the langchain fork and we work with the python files of the `langchain` repo.

If you want to use files from different repo, change `root_dir` to the root dir of your repo.
"""

# !ls "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/libs"


root_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain"
extensions = ["*.md", "*.mdx", "*.rst"]
docs_limit = 20

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        # Use fnmatch to match the extensions patterns
        if any(fnmatch.fnmatch(file, ext) for ext in extensions) and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(
                    dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                pass
# Limit docs to the first n if we collect more than that
docs = docs[:docs_limit]
print(f"{len(docs)} documents loaded.")

"""
Then, chunk the files
"""


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")

"""
Then embed chunks and upload them to the DeepLake.

This can take several minutes.
"""


embeddings = OllamaEmbeddings(model="nomic-embed-text")


# username = "<USERNAME_OR_ORG>"

db = DeepLake.from_documents(
    texts, embeddings, dataset_path=DATASET_PATH, overwrite=True
)

"""
`Optional`: You can also use Deep Lake's Managed Tensor Database as a hosting service and run queries there. In order to do so, it is necessary to specify the runtime parameter as {'tensor_db': True} during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.
"""


"""
### Question Answering
First load the dataset, construct the retriever, then construct the Conversational Chain
"""

db = DeepLake(
    dataset_path=DATASET_PATH,
    read_only=True,
    embedding=embeddings,
)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 20
# retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 20

"""
You can also specify user defined functions using [Deep Lake filters](https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter)
"""


def filter(x):
    if "something" in x["text"].data()["value"]:
        return False

    metadata = x["metadata"].data()["value"]
    return "only_this" in metadata["source"] or "also_that" in metadata["source"]


model = ChatOllama(
    model="llama3.1"
)  # 'ada' 'gpt-3.5-turbo-0613' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "What is the class hierarchy?",
    "What classes are derived from the Chain class?",
    "What kind of retrievers does LangChain have?",
]
chat_history = []
qa_dict = {}

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    qa_dict[question] = result["answer"]
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")

qa_dict

print(qa_dict["What is the class hierarchy?"])

print(qa_dict["What classes are derived from the Chain class?"])

print(qa_dict["What kind of retrievers does LangChain have?"])

logger.info("\n\n[DONE]", bright=True)
