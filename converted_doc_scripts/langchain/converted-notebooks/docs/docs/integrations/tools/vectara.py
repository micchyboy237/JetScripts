from jet.adapters.langchain.chat_ollama.chat_models import ChatOllama
from jet.logger import logger
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSerializable
from langchain_core.messages import HumanMessage
from langchain_vectara import Vectara
from langchain_vectara.tools import (
VectaraAddFiles,
VectaraIngest,
VectaraRAG,
VectaraSearch,
)
from langchain_vectara.vectorstores import (
ChainReranker,
CorpusConfig,
CustomerSpecificReranker,
File,
GenerationConfig,
MmrReranker,
SearchConfig,
VectaraQueryConfig,
)
from langgraph.prebuilt import create_react_agent
import json
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
# Vectara

[Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications. For more [details](../providers/vectara.ipynb).


[Vectara](https://vectara.com/)  provides several tools that can be used with LangChain.
- **VectaraSearch**: For semantic search over your corpus
- **VectaraRAG**: For generating summaries using RAG
- **VectaraIngest**: For ingesting documents into your corpus
- **VectaraAddFiles**: For uploading the files


## Setup

To use the `Vectara Tools` you first need to install the partner package.
"""
logger.info("# Vectara")

# !uv pip install -U pip && uv pip install -qU langchain-vectara langgraph

"""
# Getting Started

To get started, use the following steps:
1. If you don't already have one, [Sign up](https://www.vectara.com/integrations/langchain) for your free Vectara trial.
2. Within your account you can create one or more corpora. Each corpus represents an area that stores text data upon ingest from input documents. To create a corpus, use the **"Create Corpus"** button. You then provide a name to your corpus as well as a description. Optionally you can define filtering attributes and apply some advanced options. If you click on your created corpus, you can see its name and corpus ID right on the top.
3. Next you'll need to create API keys to access the corpus. Click on the **"Access Control"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name, and choose whether you want query-only or query+index for your key. Click "Create" and you now have an active API key. Keep this key confidential. 

To use LangChain with Vectara, you'll need to have these two values: `corpus_key` and `api_key`.
You can provide `VECTARA_API_KEY` to LangChain in two ways:

## Instantiation

1. Include in your environment these two variables: `VECTARA_API_KEY`.

#    For example, you can set these variables using os.environ and getpass as follows:

```python
# import getpass

# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
```

2. Add them to the `Vectara` vectorstore constructor:

```python
vectara = Vectara(
    vectara_api_key=vectara_api_key
)
```

In this notebook we assume they are provided in the environment.
"""
logger.info("# Getting Started")


os.environ["VECTARA_API_KEY"] = "<VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "<VECTARA_CORPUS_KEY>"
# os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"


vectara = Vectara(vectara_api_key=os.getenv("VECTARA_API_KEY"))

"""
First we load the state-of-the-union text into Vectara.

Note that we use the `VectaraAddFiles` tool which does not require any local processing or chunking - Vectara receives the file content and performs all the necessary pre-processing, chunking and embedding of the file into its knowledge store.

In this case it uses a .txt file but the same works for many other [file types](https://docs.vectara.com/docs/api-reference/indexing-apis/file-upload/file-upload-filetypes).
"""
logger.info("First we load the state-of-the-union text into Vectara.")

corpus_key = os.getenv("VECTARA_CORPUS_KEY")

add_files_tool = VectaraAddFiles(
    name="add_files_tool",
    description="Upload files about state of the union",
    vectorstore=vectara,
    corpus_key=corpus_key,
)

file_obj = File(
    file_path="../document_loaders/example_data/state_of_the_union.txt",
    metadata={"source": "text_file"},
)
add_files_tool.run({"files": [file_obj]})

"""
## Vectara RAG (retrieval augmented generation)

We now create a `VectaraQueryConfig` object to control the retrieval and summarization options:
* We enable summarization, specifying we would like the LLM to pick the top 7 matching chunks and respond in English

Using this configuration, let's create a LangChain tool `VectaraRAG` object that encpasulates the full Vectara RAG pipeline:
"""
logger.info("## Vectara RAG (retrieval augmented generation)")

generation_config = GenerationConfig(
    max_used_search_results=7,
    response_language="eng",
    generation_preset_name="vectara-summary-ext-24-05-med-omni",
    enable_factual_consistency_score=True,
)
search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key=corpus_key)],
    limit=25,
    reranker=ChainReranker(
        rerankers=[
            CustomerSpecificReranker(reranker_id="rnk_272725719", limit=100),
            MmrReranker(diversity_bias=0.2, limit=100),
        ]
    ),
)

config = VectaraQueryConfig(
    search=search_config,
    generation=generation_config,
)

query_str = "what did Biden say?"

vectara_rag_tool = VectaraRAG(
    name="rag-tool",
    description="Get answers about state of the union",
    vectorstore=vectara,
    corpus_key=corpus_key,
    config=config,
)

"""
## Invocation
"""
logger.info("## Invocation")

vectara_rag_tool.run(query_str)

"""
## Vectara as a langchain retriever

The `VectaraSearch` tool can be used just as a retriever. 

In this case, it behaves just like any other LangChain retriever. The main use of this mode is for semantic search, and in this case we disable summarization:
"""
logger.info("## Vectara as a langchain retriever")

search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key=corpus_key)],
    limit=25,
    reranker=ChainReranker(
        rerankers=[
            CustomerSpecificReranker(reranker_id="rnk_272725719", limit=100),
            MmrReranker(diversity_bias=0.2, limit=100),
        ]
    ),
)

search_tool = VectaraSearch(
    name="Search tool",
    description="Search for information about state of the union",
    vectorstore=vectara,
    corpus_key=corpus_key,
    search_config=search_config,
)

search_tool.run({"query": "What did Biden say?"})

"""
## Chaining with Vectara tools

You can chain Vectara tools with other LangChain components. The example shows how to:
- Set up a ChatOllama model for additional processing
- Create a custom prompt template for specific summarization needs
- Chain multiple components together using LangChain's Runnable interface
- Process and format the JSON response from Vectara
"""
logger.info("## Chaining with Vectara tools")


llm = ChatOllama(model="llama3.2")

template = """
Based on the following information from the State of the Union address:

{rag_result}

Please provide a concise summary that focuses on the key points mentioned.
If there are any specific numbers or statistics, be sure to include them.
"""
prompt = ChatPromptTemplate.from_template(template)


def get_rag_result(query: str) -> str:
    result = vectara_rag_tool.run(query)
    result_dict = json.loads(result)
    return result_dict["summary"]


chain: RunnableSerializable = (
    {"rag_result": get_rag_result} | prompt | llm | StrOutputParser()
)

chain.invoke("What were the key economic points in Biden's speech?")

"""
## Use within an agent

The code below demonstrates how to use Vectara tools with LangChain to create an agent.
"""
logger.info("## Use within an agent")



tools = [vectara_rag_tool]
llm = ChatOllama(model="llama3.2")

agent_executor = create_react_agent(llm, tools)

question = (
    "What is an API key? What is a JWT token? When should I use one or the other?"
)
input_data = {"messages": [HumanMessage(content=question)]}


agent_executor.invoke(input_data)

"""
## VectaraIngest Example

The `VectaraIngest` tool allows you to directly ingest text content into your Vectara corpus. This is useful when you have text content that you want to add to your corpus without having to create a file first.

Here's an example of how to use it:
"""
logger.info("## VectaraIngest Example")

ingest_tool = VectaraIngest(
    name="ingest_tool",
    description="Add new documents about planets",
    vectorstore=vectara,
    corpus_key=corpus_key,
)

texts = ["Mars is a red planet.", "Venus has a thick atmosphere."]

metadatas = [{"type": "planet Mars"}, {"type": "planet Venus"}]

ingest_tool.run(
    {
        "texts": texts,
        "metadatas": metadatas,
        "doc_metadata": {"test_case": "langchain tool"},
    }
)

"""
## API reference

For details checkout implementation of Vectara [tools](https://github.com/vectara/langchain-vectara/blob/main/libs/vectara/langchain_vectara/tools.py).
"""
logger.info("## API reference")


logger.info("\n\n[DONE]", bright=True)