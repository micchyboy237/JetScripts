from google.ai.generativelanguage import (
GenerateAnswerRequest,
HarmCategory,
SafetySetting,
)
from google.oauth2 import service_account
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Response
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import (
StepDecomposeQueryTransform,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.indices.managed.google import GoogleIndex
from llama_index.llms.gemini import Gemini
from llama_index.response_synthesizers.google import GoogleTextSynthesizer
from llama_index.vector_stores.google import GoogleVectorStore
from llama_index.vector_stores.google import set_google_config
from random import randrange
from typing import Iterable
import llama_index.core.vector_stores.google.generativeai.genai_extension as genaix
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/managed/GoogleDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google Generative Language Semantic Retriever

In this notebook, we will show you how to get started quickly with using Google's Generative Language Semantic Retriever, which offers specialized embedding models for high quality retrieval and a tuned model for producing grounded output with customizable safety settings. We will also show you some advanced examples on how to combine the power of LlamaIndex and this unique offering from Google.

## Installation
"""
logger.info("# Google Generative Language Semantic Retriever")

# %pip install llama-index-llms-gemini
# %pip install llama-index-vector-stores-google
# %pip install llama-index-indices-managed-google
# %pip install llama-index-response-synthesizers-google

# %pip install llama-index
# %pip install "google-ai-generativelanguage>=0.4,<=1.0"

"""
### Google Authentication Overview

The Google Semantic Retriever API lets you perform semantic search on your own data. Since it's **your data**, this needs stricter access controls than API keys. Authenticate with OAuth with service accounts or through your user credentials (example in the bottom of the notebook).

This quickstart uses a simplified authentication approach meant for a testing environment, and service account setup are typically easier to start from. Demo recording for authenticating using service accounts: [Demo](https://drive.google.com/file/d/199LzrdhuuiordS15MJAxVrPKAwEJGPOh/view?usp=sharing).

For a production environment, learn about [authentication and authorization](https://developers.google.com/workspace/guides/auth-overview) before choosing the [access credentials](https://developers.google.com/workspace/guides/create-credentials#choose_the_access_credential_that_is_right_for_you) that are appropriate for your app.

**Note**: At this time, the Google Generative AI Semantic Retriever API is [only available in certain regions](https://ai.google.dev/available_regions).

### Setup OAuth using service accounts

Follow the steps below to setup OAuth using service accounts:


1. Enable the [Generative Language API](https://console.cloud.google.com/flows/enableapi?apiid=generativelanguage.googleapis.com).

2. Create the Service Account by following the [documentation](https://developers.google.com/identity/protocols/oauth2/service-account#creatinganaccount).

 * After creating the service account, generate a service account key.

3. Upload your service account file by using the file icon on the left sidebar, then the upload icon, as shown in the screenshot below.

 * Rename the uploaded file to `service_account_key.json` or change the variable `service_account_file_name` in the code below.

<img width=400 src="https://developers.generativeai.google/tutorials/images/colab_upload.png">
"""
logger.info("### Google Authentication Overview")

# %pip install google-auth-oauthlib


credentials = service_account.Credentials.from_service_account_file(
    "service_account_key.json",
    scopes=[
        "https://www.googleapis.com/auth/generative-language.retriever",
    ],
)
set_google_config(auth_credentials=credentials)

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Setup

First, let's create some helper functions behind the scene.
"""
logger.info("## Setup")



LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX = f"llama-index-colab"
SESSION_CORPUS_ID_PREFIX = (
    f"{LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX}-{randrange(1000000)}"
)


def corpus_id(num_id: int) -> str:
    return f"{SESSION_CORPUS_ID_PREFIX}-{num_id}"


SESSION_CORPUS_ID = corpus_id(1)


def list_corpora() -> Iterable[genaix.Corpus]:
    client = genaix.build_semantic_retriever()
    yield from genaix.list_corpora(client=client)


def delete_corpus(*, corpus_id: str) -> None:
    client = genaix.build_semantic_retriever()
    genaix.delete_corpus(corpus_id=corpus_id, client=client)


def cleanup_colab_corpora():
    for corpus in list_corpora():
        if corpus.corpus_id.startswith(LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX):
            try:
                delete_corpus(corpus_id=corpus.corpus_id)
                logger.debug(f"Deleted corpus {corpus.corpus_id}.")
            except Exception:
                pass


cleanup_colab_corpora()

"""
## Basic Usage

A `corpus` is a collection of `document`s. A `document` is a body of text that is broken into `chunk`s.
"""
logger.info("## Basic Usage")


index = GoogleIndex.create_corpus(
    corpus_id=SESSION_CORPUS_ID, display_name="My first corpus!"
)
logger.debug(f"Newly created corpus ID is {index.corpus_id}.")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
index.insert_documents(documents)

"""
Let's check that what we've ingested.
"""
logger.info("Let's check that what we've ingested.")

for corpus in list_corpora():
    logger.debug(corpus)

"""
Let's ask the index a question.
"""
logger.info("Let's ask the index a question.")

query_engine = index.as_query_engine()
response = query_engine.query("What did Paul Graham do growing up?")
assert isinstance(response, Response)

logger.debug(f"Response is {response.response}")

for cited_text in [node.text for node in response.source_nodes]:
    logger.debug(f"Cited text: {cited_text}")

if response.metadata:
    logger.debug(
        f"Answerability: {response.metadata.get('answerable_probability', 0)}"
    )

"""
## Creating a Corpus

There are various ways to create a corpus.

```python
# The Google server will provide a corpus ID for you.
index = GoogleIndex.create_corpus(display_name="My first corpus!")
logger.debug(index.corpus_id)

# You can also provide your own corpus ID. However, this ID needs to be globally
# unique. You will get an exception if someone else has this ID already.
index = GoogleIndex.create_corpus(
    corpus_id="my-first-corpus", display_name="My first corpus!"
)

# If you do not provide any parameter, Google will provide ID and a default
# display name for you.
index = GoogleIndex.create_corpus()
```

## Reusing a Corpus

Corpora you created persists on the Google servers under your account.
You can use its ID to get a handle back.
Then, you can query it, add more document to it, etc.
"""
logger.info("## Creating a Corpus")

index = GoogleIndex.from_corpus(corpus_id=SESSION_CORPUS_ID)

query_engine = index.as_query_engine()
response = query_engine.query("Which company did Paul Graham build?")
assert isinstance(response, Response)

logger.debug(f"Response is {response.response}")

"""
## Listing and Deleting Corpora

See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

## Loading Documents

Many node parsers and text splitters in LlamaIndex automatically add to each node a *source_node* to associate it to a file, e.g.

```python
    relationships={
        NodeRelationship.SOURCE: RelatedNodeInfo(
            node_id="abc-123",
            metadata={"file_name": "Title for the document"},
        )
    },
```

Both `GoogleIndex` and `GoogleVectorStore` recognize this source node,
and will automatically create documents under your corpus on the Google servers.

In case you are writing your own chunker, you should supply this source node relationship too like below:
"""
logger.info("## Listing and Deleting Corpora")


index = GoogleIndex.from_corpus(corpus_id=SESSION_CORPUS_ID)
index.insert_nodes(
    [
        TextNode(
            text="It was the best of times.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="123",
                    metadata={"file_name": "Tale of Two Cities"},
                )
            },
        ),
        TextNode(
            text="It was the worst of times.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="123",
                    metadata={"file_name": "Tale of Two Cities"},
                )
            },
        ),
        TextNode(
            text="Bugs Bunny: Wassup doc?",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="456",
                    metadata={"file_name": "Bugs Bunny Adventure"},
                )
            },
        ),
    ]
)

"""
If your nodes do not have a source node, then Google server will put your nodes in a default document under your corpus.

## Listing and Deleting Documents

See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

## Querying Corpus

Google's query engine is backed by a specially tuned LLM that grounds its response based on retrieved passages. For each response, an *answerability probability* is returned to indicate how confident the LLM was in answering the question from the retrieved passages.

Furthermore, Google's query engine supports *answering styles*, such as `ABSTRACTIVE` (succint but abstract), `EXTRACTIVE` (very brief and extractive) and `VERBOSE` (extra details).

The engine also supports *safety settings*.
"""
logger.info("## Listing and Deleting Documents")


index = GoogleIndex.from_corpus(corpus_id=SESSION_CORPUS_ID)
query_engine = index.as_query_engine(
    temperature=0.2,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
    safety_setting=[
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_VIOLENCE,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ],
)

response = query_engine.query("What was Bugs Bunny's favorite saying?")
logger.debug(response)

"""
See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

## Interpreting the Response
"""
logger.info("## Interpreting the Response")


response = query_engine.query("What were Paul Graham's achievements?")
assert isinstance(response, Response)

logger.debug(f"Response is {response.response}")

for cited_text in [node.text for node in response.source_nodes]:
    logger.debug(f"Cited text: {cited_text}")

if response.metadata:
    logger.debug(
        f"Answerability: {response.metadata.get('answerable_probability', 0)}"
    )

"""
## Advanced RAG

The `GoogleIndex` is built based on `GoogleVectorStore` and `GoogleTextSynthesizer`.
These components can be combined with other powerful constructs in LlamaIndex to produce advanced RAG applications.

Below we show a few examples.

## Setup

First, you need an API key. Get one from [AI Studio](https://makersuite.google.com/app/apikey).
"""
logger.info("## Advanced RAG")


GEMINI_API_KEY = ""  # @param {type:"string"}
gemini = Gemini(api_key=GEMINI_API_KEY)

"""
### Reranker + Google Retriever

Converting content into vectors is a lossy process. LLM-based Reranking
remediates this by reranking the retrieved content using LLM, which has higher
fidelity because it has access to both the actual query and the passage.
"""
logger.info("### Reranker + Google Retriever")


store = GoogleVectorStore.from_corpus(corpus_id=SESSION_CORPUS_ID)
index = VectorStoreIndex.from_vector_store(
    vector_store=store,
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.2,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
)
reranker = LLMRerank(
    top_n=10,
    llm=gemini,
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    ),
    node_postprocessors=[reranker],
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("What were Paul Graham's achievements?")
logger.debug(response)

"""
### Multi-Query + Google Retriever

Sometimes, a user's query can be too complex. You may get better retrieval result if you break down the original query into smaller, better focused queries.
"""
logger.info("### Multi-Query + Google Retriever")


store = GoogleVectorStore.from_corpus(corpus_id=SESSION_CORPUS_ID)
index = VectorStoreIndex.from_vector_store(
    vector_store=store,
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.2,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
)
single_step_query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_synthesizer=response_synthesizer,
)
step_decompose_transform = StepDecomposeQueryTransform(
    llm=gemini,
    verbose=True,
)
query_engine = MultiStepQueryEngine(
    query_engine=single_step_query_engine,
    query_transform=step_decompose_transform,
    response_synthesizer=response_synthesizer,
    index_summary="Ask me anything.",
    num_steps=6,
)

response = query_engine.query("What were Paul Graham's achievements?")
logger.debug(response)

"""
### HyDE + Google Retriever

When you can write prompt that would produce fake answers that share many traits
with the real answer, you can try HyDE!
"""
logger.info("### HyDE + Google Retriever")


store = GoogleVectorStore.from_corpus(corpus_id=SESSION_CORPUS_ID)
index = VectorStoreIndex.from_vector_store(
    vector_store=store,
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.2,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
)
base_query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_synthesizer=response_synthesizer,
)
hyde = HyDEQueryTransform(
    llm=gemini,
    include_original=False,
)
hyde_query_engine = TransformQueryEngine(base_query_engine, hyde)

response = query_engine.query("What were Paul Graham's achievements?")
logger.debug(response)

"""
### Multi-Query + Reranker + HyDE + Google Retriever

Or combine them all!
"""
logger.info("### Multi-Query + Reranker + HyDE + Google Retriever")

store = GoogleVectorStore.from_corpus(corpus_id=SESSION_CORPUS_ID)
index = VectorStoreIndex.from_vector_store(
    vector_store=store,
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.2, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)

reranker = LLMRerank(
    top_n=10,
    llm=gemini,
)
single_step_query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
    response_synthesizer=response_synthesizer,
)

hyde = HyDEQueryTransform(
    llm=gemini,
    include_original=False,
)
hyde_query_engine = TransformQueryEngine(single_step_query_engine, hyde)

step_decompose_transform = StepDecomposeQueryTransform(
    llm=gemini, verbose=True
)
query_engine = MultiStepQueryEngine(
    query_engine=hyde_query_engine,
    query_transform=step_decompose_transform,
    response_synthesizer=response_synthesizer,
    index_summary="Ask me anything.",
    num_steps=6,
)

response = query_engine.query("What were Paul Graham's achievements?")
logger.debug(response)

"""
## Cleanup corpora created in the colab
"""
logger.info("## Cleanup corpora created in the colab")

cleanup_colab_corpora()

"""
## Appendix: Setup OAuth with user credentials

Please follow [OAuth Quickstart](https://developers.generativeai.google/tutorials/oauth_quickstart) to setup OAuth using user credentials. Below are overview of steps from the documentation that are required.

1. Enable the `Generative Language API`: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#1_enable_the_api)

1. Configure the OAuth consent screen: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#2_configure_the_oauth_consent_screen)

1. Authorize credentials for a desktop application: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#3_authorize_credentials_for_a_desktop_application)
  * If you want to run this notebook in Colab start by uploading your
`client_secret*.json` file using the "File > Upload" option.

 * Rename the uploaded file to `client_secret.json` or change the variable `client_file_name` in the code below.

<img width=400 src="https://developers.generativeai.google/tutorials/images/colab_upload.png">
"""
logger.info("## Appendix: Setup OAuth with user credentials")

project_name = "TODO-your-project-name"  #  @param {type:"string"}
email = "TODO-your-email@gmail.com"  #  @param {type:"string"}
client_file_name = "client_secret.json"

# !gcloud config set project $project_name
# !gcloud config set account $email

# !gcloud auth application-default login --no-browser --client-id-file=$client_file_name --scopes="https://www.googleapis.com/auth/generative-language.retriever,https://www.googleapis.com/auth/cloud-platform"

"""
This will provide you with a URL, which you should enter into your local browser.
Follow the instruction to complete the authentication and authorization.
"""
logger.info("This will provide you with a URL, which you should enter into your local browser.")

logger.info("\n\n[DONE]", bright=True)