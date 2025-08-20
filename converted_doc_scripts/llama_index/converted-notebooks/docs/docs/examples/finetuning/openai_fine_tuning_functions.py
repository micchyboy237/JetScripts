import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from json import JSONDecodeError
from llama_index.core import Document
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.finetuning import MLXFinetuneEngine
from llama_index.finetuning.callbacks import MLXFineTuningHandler
from llama_index.program.openai import MLXPydanticProgram
from pathlib import Path
from pydantic import BaseModel
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio
from tqdm.notebook import tqdm
from typing import List
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/openai_fine_tuning_functions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Fine Tuning with Function Calling

In this notebook, we walk through how to fine-tune gpt-3.5-turbo with function calls. The primary use case here is structured data extraction. Our main focus is distilling GPT-4 outputs to help improve gpt-3.5-turbo function calling capabilities.

We will walk through some examples, from simple to advanced:
1. Fine-tuning on some toy messages/structured outputs logged through our MLX Pydantic Program object.
2. Fine-tuning on context-augmented queries/structured outputs over an entire document corpus. Use this in a RAG system.
"""
logger.info("# Fine Tuning with Function Calling")

# %pip install llama-index-finetuning
# %pip install llama-index-llms-ollama
# %pip install llama-index-finetuning-callbacks
# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-program-openai

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Fine-tuning Using GPT-4 Pydantic Programs

In this section we show how to log inputs/outputs through our low-level Pydantic Program module. We use that dataset to fine-tune an LLM.

### Defining Pydantic Model + Program

Here, we define the GPT-4 powered function calling program that will generate structured outputs into a Pydantic object (an Album).
"""
logger.info("## Fine-tuning Using GPT-4 Pydantic Programs")



class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]


finetuning_handler = MLXFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", callback_manager=callback_manager)


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = MLXPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=False,
)

"""
### Log Inputs/Outputs

We define some sample movie names as inputs and log the outputs through the function calling program.
"""
logger.info("### Log Inputs/Outputs")

movie_names = [
    "The Shining",
    "The Departed",
    "Titanic",
    "Goodfellas",
    "Pretty Woman",
    "Home Alone",
    "Caged Fury",
    "Edward Scissorhands",
    "Total Recall",
    "Ghost",
    "Tremors",
    "RoboCop",
    "Rocky V",
]


for movie_name in tqdm(movie_names):
    output = program(movie_name=movie_name)
    logger.debug(output.json())

finetuning_handler.save_finetuning_events("mock_finetune_songs.jsonl")

# !cat mock_finetune_songs.jsonl

"""
### Fine-tune on the Dataset

We now define a fine-tuning engine and fine-tune on the mock dataset.
"""
logger.info("### Fine-tune on the Dataset")


finetune_engine = MLXFinetuneEngine(
    "gpt-3.5-turbo",
    "mock_finetune_songs.jsonl",
    validate_json=False,  # openai validate json code doesn't support function calling yet
)

finetune_engine.finetune()

finetune_engine.get_current_job()

"""
### Try it Out! 

We obtain the fine-tuned LLM and use it with the Pydantic program.
"""
logger.info("### Try it Out!")

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)

ft_program = MLXPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=ft_llm,
    verbose=False,
)

ft_program(movie_name="Goodfellas")

"""
## Fine-tuning Structured Outputs through a RAG System

A use case of function calling is to get structured outputs through a RAG system.

Here we show how to create a training dataset of context-augmented inputs + structured outputs over an unstructured document. We can then fine-tune the LLM and plug it into a RAG system to perform retrieval + output extraction.
"""
logger.info("## Fine-tuning Structured Outputs through a RAG System")

# !mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"



class Citation(BaseModel):
    """Citation class."""

    author: str = Field(
        ..., description="Inferred first author (usually last name"
    )
    year: int = Field(..., description="Inferred year")
    desc: str = Field(
        ...,
        description=(
            "Inferred description from the text of the work that the author is"
            " cited for"
        ),
    )


class Response(BaseModel):
    """List of author citations.

    Extracted over unstructured text.

    """

    citations: List[Citation] = Field(
        ...,
        description=(
            "List of author citations (organized by author, year, and"
            " description)."
        ),
    )

"""
### Load Data + Setup
"""
logger.info("### Load Data + Setup")

# from llama_index.readers.file import PyMuPDFReader

# loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs0])
metadata = {
    "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
}
docs = [Document(text=doc_text, metadata=metadata)]

chunk_size = 1024
node_parser = SentenceSplitter(chunk_size=chunk_size)
nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)


finetuning_handler = MLXFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

Settings.chunk_size = chunk_size

gpt_4_llm = MLX(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.3, callback_manager=callback_manager
)

gpt_35_llm = MLX(
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    temperature=0.3,
    callback_manager=callback_manager,
)

eval_llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)

"""
### Generate Dataset

Here we show how to generate a training dataset over these unstructured chunks/nodes.

We generate questions to extract citations over different context. We run these questions through a GPT-4 RAG pipeline, extract structured outputs, and log inputs/outputs.
"""
logger.info("### Generate Dataset")



fp = open(f"{GENERATED_DIR}/qa_pairs.jsonl", "w")

question_gen_prompt = PromptTemplate(
    """
{query_str}

Context:
{context_str}

Questions:
"""
)

question_gen_query = """\
Snippets from a research paper is given below. It contains citations.
Please generate questions from the text asking about these citations.

For instance, here are some sample questions:
Which citations correspond to related works on transformer models?
Tell me about authors that worked on advancing RLHF.
Can you tell me citations corresponding to all computer vision works? \
"""

qr_pairs = []
node_questions_tasks = []
for idx, node in enumerate(nodes[:39]):
    num_questions = 1  # change this number to increase number of nodes
    dataset_generator = DatasetGenerator(
        [node],
        question_gen_query=question_gen_query,
        text_question_template=question_gen_prompt,
        llm=eval_llm,
        metadata_mode="all",
        num_questions_per_chunk=num_questions,
    )

    task = dataset_generator.agenerate_questions_from_nodes(num=num_questions)
    node_questions_tasks.append(task)
async def run_async_code_c35c4b08():
    async def run_async_code_7e6ba558():
        node_questions_lists = await tqdm_asyncio.gather(*node_questions_tasks)
        return node_questions_lists
    node_questions_lists = asyncio.run(run_async_code_7e6ba558())
    logger.success(format_json(node_questions_lists))
    return node_questions_lists
node_questions_lists = asyncio.run(run_async_code_c35c4b08())
logger.success(format_json(node_questions_lists))

node_questions_lists


gpt4_index = VectorStoreIndex(nodes=nodes)
gpt4_query_engine = gpt4_index.as_query_engine(
    output_cls=Response, similarity_top_k=1, llm=gpt_4_llm
)


for idx, node in enumerate(tqdm(nodes[:39])):
    node_questions_0 = node_questions_lists[idx]
    for question in node_questions_0:
        try:
            gpt4_query_engine.query(question)
        except Exception as e:
            logger.debug(f"Error for question {question}, {repr(e)}")
            pass

finetuning_handler.save_finetuning_events("llama2_citation_events.jsonl")

"""
### Setup Fine-tuning

We kick off fine-tuning over the generated dataset.
"""
logger.info("### Setup Fine-tuning")


finetune_engine = MLXFinetuneEngine(
    "gpt-3.5-turbo",
    "llama2_citation_events.jsonl",
    validate_json=False,  # openai validate json code doesn't support function calling yet
)

finetune_engine.finetune()

finetune_engine.get_current_job()

"""
### Use within RAG Pipeline

Let's plug the fine-tuned LLM into a full RAG pipeline that outputs structured outputs.
"""
logger.info("### Use within RAG Pipeline")

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)


vector_index = VectorStoreIndex(nodes=nodes)
query_engine = vector_index.as_query_engine(
    output_cls=Response, similarity_top_k=1, llm=ft_llm
)

base_index = VectorStoreIndex(nodes=nodes)
base_query_engine = base_index.as_query_engine(
    output_cls=Response, similarity_top_k=1, llm=gpt_35_llm
)

query_str = """\
Which citation is used to measure the truthfulness of Llama 2? \
"""


response = query_engine.query(query_str)
logger.debug(str(response))

base_response = base_query_engine.query(query_str)
logger.debug(str(base_response))

logger.debug(response.source_nodes[0].get_content())

gpt4_response = gpt4_query_engine.query(query_str)
logger.debug(str(gpt4_response))

logger.info("\n\n[DONE]", bright=True)