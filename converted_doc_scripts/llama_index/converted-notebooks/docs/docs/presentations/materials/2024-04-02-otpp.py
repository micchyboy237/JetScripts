from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.program.openai import MLXPydanticProgram
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
import json
import os
import qdrant_client
import shutil


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
# LLMs and LlamaIndex ◦ April 2 2024 ◦ Ontario Teacher's Pension Plan
"""
logger.info("# LLMs and LlamaIndex ◦ April 2 2024 ◦ Ontario Teacher's Pension Plan")



"""
![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/title.excalidraw.svg)

![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/framework.excalidraw.svg)

#### Notebook Setup & Dependency Installation
"""
logger.info("#### Notebook Setup & Dependency Installation")

# %pip install llama-index-vector-stores-qdrant -q

# import nest_asyncio

# nest_asyncio.apply()

# !mkdir data
# !wget "https://www.otpp.com/content/dam/otpp/documents/reports/2023-ar/otpp-2023-annual-report-eng.pdf" -O "./data/otpp-2023-annual-report-eng.pdf"

"""
## Motivation

![Motivation Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/motivation.excalidraw.svg)
"""
logger.info("## Motivation")


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
response = llm.complete("What is Ontario Teacher's Pension Plan all about?")

logger.debug(response)

response = llm.complete(
    "According to the 2023 annual report, how many billions of dollars in net assets does Ontario Teacher's Pension Plan hold?"
)

logger.debug(response)

response = llm.complete(
    "According to the 2023 annual report, what is the 10-year total-fund net return?"
)

logger.debug(response)

"""
## Basic RAG in 3 Steps

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/subheading.excalidraw.svg)


1. Build external knowledge (i.e., uploading updated data sources)
2. Retrieve
3. Augment and Generate

## 1. Build External Knowledge

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step1.excalidraw.svg)
"""
logger.info("## Basic RAG in 3 Steps")

"""Load the data.

With llama-index, before any transformations are applied,
data is loaded in the `Document` abstraction, which is
a container that holds the text of the document.
"""


loader = SimpleDirectoryReader(input_dir="./data")
documents = loader.load_data()

documents[0].text[:1000]

"""Chunk, Encode, and Store into a Vector Store.

To streamline the process, we can make use of the IngestionPipeline
class that will apply your specified transformations to the
Document's.
"""


client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        MLXEmbedding(),
    ],
    vector_store=vector_store,
)
_nodes = pipeline.run(documents=documents, num_workers=4)

_nodes[0].text

"""Create a llama-index... wait for it... Index.

After uploading your encoded documents into your vector
store of choice, you can connect to it with a VectorStoreIndex
which then gives you access to all of the llama-index functionality.
"""


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
## 2. Retrieve Against A Query

![Step2 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step2.excalidraw.svg)
"""
logger.info("## 2. Retrieve Against A Query")

"""Retrieve relevant documents against a query.

With our Index ready, we can now query it to
retrieve the most relevant document chunks.
"""

retriever = index.as_retriever(similarity_top_k=2)
retrieved_nodes = retriever.retrieve(
    "According to the 2023 annual report, what is the 10-year total-fund net return?"
)

logger.debug(retrieved_nodes[0].text[:500])

"""
## 3. Generate Final Response

![Step3 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step3.excalidraw.svg)
"""
logger.info("## 3. Generate Final Response")

"""Context-Augemented Generation.

With our Index ready, we can create a QueryEngine
that handles the retrieval and context augmentation
in order to get the final response.
"""

query_engine = index.as_query_engine()

logger.debug(
    query_engine.get_prompts()[
        "response_synthesizer:text_qa_template"
    ].default_template.template
)

response = query_engine.query(
    "According to the 2023 annual report, what is the 10-year total-fund net return?"
)
logger.debug(response)

"""
## Beyond Basic RAG: Improved PDF Parsing with LlamaParse

To use LlamaParse, you first need to obtain an API Key. Visit [llamacloud.ai](https://cloud.llamaindex.ai/login) to login (or sign-up) and get an api key.
"""
logger.info("## Beyond Basic RAG: Improved PDF Parsing with LlamaParse")

api_key = "<FILL-IN>"

"""
### The default pdf reader (PyPDF), like many out-of-the box pdf parsers struggle on complex PDF docs.
"""
logger.info("### The default pdf reader (PyPDF), like many out-of-the box pdf parsers struggle on complex PDF docs.")

response = query_engine.query(
    "How many board meetings did Steve McGirr, Chair of the Board, attend?"
)
logger.debug(response)

response = query_engine.query(
    "What percentage of board members identify as women?"
)
logger.debug(response)

response = query_engine.query(
    "What is the total investment percentage in Canada as of December 31, 2023?"
)
logger.debug(response)

"""
### Improved PDF Parsing using LlamaParse
"""
logger.info("### Improved PDF Parsing using LlamaParse")



parser = LlamaParse(result_type="markdown")
md_documents = parser.load_data(
    file_path="./data/otpp-2023-annual-report-eng.pdf"
)

with open("./mds/parsed.md", "w") as f:
    f.write(md_documents[0].text)

md_node_parser = MarkdownElementNodeParser(
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
    num_workers=3,
    include_metadata=True,
)
md_nodes = md_node_parser.get_nodes_from_documents(md_documents)

llama_parse_index = VectorStoreIndex.from_documents(md_documents)
llama_parse_query_engine = llama_parse_index.as_query_engine()

response = llama_parse_query_engine.query(
    "How many board meetings did Steve McGirr, Chair of the Board, attend?"
)
logger.debug(response)

response = llama_parse_query_engine.query(
    "What percentage of board members identify as women?"
)
logger.debug(response)

response = llama_parse_query_engine.query(
    "What is the total investment percentage in Canada as of December 31, 2023?"
)
logger.debug(response)

"""
## In Summary

- LLMs as powerful as they are, don't perform too well with knowledge-intensive tasks (domain specific, updated data, long-tail)
- Context augmentation has been shown (in a few studies) to outperform LLMs without augmentation
- In this notebook, we showed one such example that follows that pattern.

## Data Extraction

![DataExtractions](https://media.licdn.com/dms/image/D4E22AQGwPmZ5RRhbyA/feedshare-shrink_1280/0/1711823067172?e=1715212800&v=beta&t=fJtksPZ3Fm-BOrKRCwa6BrYyuxlNFDuop3ZSopMl53M)
"""
logger.info("## In Summary")



"""
### Leadership Team
"""
logger.info("### Leadership Team")

class LeadershipTeam(BaseModel):
    """Data model for leadership team."""

    ceo: str = Field(description="The CEO")
    coo: str = Field(description="The Chief Operating Officer")
    cio: str = Field(description="Chief Investment Officer")
    chief_pension_officer: str = Field(description="Chief Pension Officer")
    chief_legal_officer: str = Field(
        description="Chief Legal & Corporate Affairs Officer"
    )
    chief_people_officer: str = Field(description="Chief People Officer")
    chief_strategy_officer: str = Field(description="Chief Strategy Officer")
    executive_managing_director: str = Field(
        description="Executive Managing Director"
    )
    chief_investment_officer: str = Field(
        description="Chief Investment Officer"
    )

prompt_template_str = """\
Here is the 2023 Annual Report for Ontario Teacher's Pension Plan:
{document_text}
Provide the names of the Leadership Team.
"""

program = MLXPydanticProgram.from_defaults(
    output_cls=LeadershipTeam,
    prompt_template_str=prompt_template_str,
    llm=MLXLlamaIndexLLMAdapter("gpt-4-turbo-preview"),
    verbose=True,
)

leadership_team = program(document_text=md_documents[0].text)

logger.debug(json.dumps(leadership_team.dict(), indent=4))

"""
# LlamaIndex Has More To Offer

- Data infrastructure that enables production-grade, advanced RAG systems
- Agentic solutions
- Newly released: `llama-index-networks`
- Enterprise offerings (alpha):
    - LlamaParse (proprietary complex PDF parser) and
    - LlamaCloud

### Useful links

[website](https://www.llamaindex.ai/) ◦ [llamahub](https://llamahub.ai) ◦ [llamaparse](https://github.com/run-llama/llama_parse) ◦ [github](https://github.com/run-llama/llama_index) ◦ [medium](https://medium.com/@llama_index) ◦ [rag-bootcamp-poster](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/final_poster.excalidraw.svg)
"""
logger.info("# LlamaIndex Has More To Offer")

logger.info("\n\n[DONE]", bright=True)