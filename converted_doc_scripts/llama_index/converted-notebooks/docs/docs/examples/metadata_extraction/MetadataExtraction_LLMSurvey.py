from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import set_global_handler
from llama_index.core.extractors import (
SummaryExtractor,
QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.response.notebook_utils import (
display_source_node,
display_response,
)
from llama_index.core.schema import MetadataMode
from llama_index.readers.web import SimpleWebPageReader
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/metadata_extraction/MetadataExtraction_LLMSurvey.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Automated Metadata Extraction for Better Retrieval + Synthesis

In this tutorial, we show you how to perform automated metadata extraction for better retrieval results.
We use two extractors: a QuestionAnsweredExtractor which generates question/answer pairs from a piece of text, and also a SummaryExtractor which extracts summaries, not only within the current text, but also within adjacent texts.

We show that this allows for "chunk dreaming" - each individual chunk can have more "holistic" details, leading to higher answer quality given retrieved results.

Our data source is taken from Eugene Yan's popular article on LLM Patterns: https://eugeneyan.com/writing/llm-patterns/

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Automated Metadata Extraction for Better Retrieval + Synthesis")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-web

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()



set_global_handler("wandb", run_args={"project": "llamaindex"})

# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Define Metadata Extractors

Here we define metadata extractors. We define two variants:
- metadata_extractor_1 only contains the QuestionsAnsweredExtractor
- metadata_extractor_2 contains both the QuestionsAnsweredExtractor as well as the SummaryExtractor
"""
logger.info("## Define Metadata Extractors")


llm = MLX(temperature=0.1, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_tokens=512)

"""
We also show how to instantiate the `SummaryExtractor` and `QuestionsAnsweredExtractor`.
"""
logger.info("We also show how to instantiate the `SummaryExtractor` and `QuestionsAnsweredExtractor`.")


node_parser = TokenTextSplitter(
    separator=" ", chunk_size=256, chunk_overlap=128
)


extractors_1 = [
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

extractors_2 = [
    SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

"""
## Load in Data, Run Extractors

We load in Eugene's essay (https://eugeneyan.com/writing/llm-patterns/) using our LlamaHub SimpleWebPageReader.

We then run our extractors.
"""
logger.info("## Load in Data, Run Extractors")



reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])

logger.debug(docs[0].get_content())

orig_nodes = node_parser.get_nodes_from_documents(docs)

nodes = orig_nodes[20:28]

logger.debug(nodes[3].get_content(metadata_mode="all"))

"""
### Run metadata extractors
"""
logger.info("### Run metadata extractors")


pipeline = IngestionPipeline(transformations=[node_parser, *extractors_1])

nodes_1 = pipeline.run(nodes=nodes, in_place=False, show_progress=True)

logger.debug(nodes_1[3].get_content(metadata_mode="all"))

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_2])

nodes_2 = pipeline.run(nodes=nodes, in_place=False, show_progress=True)

"""
### Visualize some sample data
"""
logger.info("### Visualize some sample data")

logger.debug(nodes_2[3].get_content(metadata_mode="all"))

logger.debug(nodes_2[1].get_content(metadata_mode="all"))

"""
## Setup RAG Query Engines, Compare Results! 

We setup 3 indexes/query engines on top of the three node variants.
"""
logger.info("## Setup RAG Query Engines, Compare Results!")


index0 = VectorStoreIndex(orig_nodes)
index1 = VectorStoreIndex(orig_nodes[:20] + nodes_1 + orig_nodes[28:])
index2 = VectorStoreIndex(orig_nodes[:20] + nodes_2 + orig_nodes[28:])

query_engine0 = index0.as_query_engine(similarity_top_k=1)
query_engine1 = index1.as_query_engine(similarity_top_k=1)
query_engine2 = index2.as_query_engine(similarity_top_k=1)

"""
### Try out some questions

In this question, we see that the naive response `response0` only mentions BLEU and ROUGE, and lacks context about other metrics.

`response2` on the other hand has all metrics within its context.
"""
logger.info("### Try out some questions")

query_str = (
    "Can you describe metrics for evaluating text generation quality, compare"
    " them, and tell me about their downsides"
)

response0 = query_engine0.query(query_str)
response1 = query_engine1.query(query_str)
response2 = query_engine2.query(query_str)

display_response(
    response0, source_length=1000, show_source=True, show_source_metadata=True
)

logger.debug(response0.source_nodes[0].node.get_content())

display_response(
    response1, source_length=1000, show_source=True, show_source_metadata=True
)

display_response(
    response2, source_length=1000, show_source=True, show_source_metadata=True
)

"""
In this next question, we ask about BERTScore/MoverScore. 

The responses are similar. But `response2` gives slightly more detail than `response0` since it has more information about MoverScore contained in the Metadata.
"""
logger.info("In this next question, we ask about BERTScore/MoverScore.")

query_str = (
    "Can you give a high-level overview of BERTScore/MoverScore + formulas if"
    " available?"
)

response0 = query_engine0.query(query_str)
response1 = query_engine1.query(query_str)
response2 = query_engine2.query(query_str)

display_response(
    response0, source_length=1000, show_source=True, show_source_metadata=True
)

display_response(
    response1, source_length=1000, show_source=True, show_source_metadata=True
)

display_response(
    response2, source_length=1000, show_source=True, show_source_metadata=True
)

response1.source_nodes[0].node.metadata

logger.info("\n\n[DONE]", bright=True)