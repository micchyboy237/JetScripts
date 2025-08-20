import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.core.evaluation.eval_utils import aget_responses
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
import os
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
# EmotionPrompt in RAG

Inspired by the "[Large Language Models Understand and Can Be Enhanced by
Emotional Stimuli](https://arxiv.org/pdf/2307.11760.pdf)" by Li et al., in this guide we show you how to evaluate the effects of emotional stimuli on your RAG pipeline:

1. Setup the RAG pipeline with a basic vector index with the core QA template.
2. Create some candidate stimuli (inspired by Fig. 2 of the paper)
3. For each candidate stimulit, prepend to QA prompt and evaluate.
"""
logger.info("# EmotionPrompt in RAG")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file pymupdf

"""
## Setup Data

We use the Llama 2 paper as the input data source for our RAG pipeline.
"""
logger.info("## Setup Data")

# !mkdir -p llama_2_data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "llama_2_data/llama2.pdf"

# from llama_index.readers.file import PyMuPDFReader

# docs0 = PyMuPDFReader().load_data("./llama_2_data/llama2.pdf")

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)

"""
## Setup Vector Index over this Data

We load this data into an in-memory vector store (embedded with MLX embeddings).

We'll be aggressively optimizing the QA prompt for this RAG pipeline.
"""
logger.info("## Setup Vector Index over this Data")


# os.environ["OPENAI_API_KEY"] = "sk-..."



Settings.llm = MLX(model="qwen3-1.7b-4bit-mini")
Settings.embed_model = MLXEmbedding(model="mxbai-embed-large")


index = VectorStoreIndex(base_nodes)

query_engine = index.as_query_engine(similarity_top_k=2)

"""
## Evaluation Setup

#### Golden Dataset

Here we load in a "golden" dataset.

**NOTE**: We pull this in from Dropbox. For details on how to generate a dataset please see our `DatasetGenerator` module.
"""
logger.info("## Evaluation Setup")

# !wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O llama2_eval_qr_dataset.json


eval_dataset = QueryResponseDataset.from_json("./llama2_eval_qr_dataset.json")

"""
#### Get Evaluator
"""
logger.info("#### Get Evaluator")



evaluator_c = CorrectnessEvaluator()

evaluator_dict = {"correctness": evaluator_c}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

"""
#### Define Correctness Eval Function
"""
logger.info("#### Define Correctness Eval Function")



async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    async def async_func_7():
        pred_responses = get_responses(
            eval_qs, query_engine, show_progress=True
        )
        return pred_responses
    pred_responses = asyncio.run(async_func_7())
    logger.success(format_json(pred_responses))

    async def async_func_11():
        eval_results = batch_runner.evaluate_responses(
            eval_qs, responses=pred_responses, reference=eval_answers
        )
        return eval_results
    eval_results = asyncio.run(async_func_11())
    logger.success(format_json(eval_results))
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness

"""
## Try Out Emotion Prompts

We pul some emotion stimuli from the paper to try out.
"""
logger.info("## Try Out Emotion Prompts")

emotion_stimuli_dict = {
    "ep01": "Write your answer and give me a confidence score between 0-1 for your answer. ",
    "ep02": "This is very important to my career. ",
    "ep03": "You'd better be sure.",
}

emotion_stimuli_dict["ep06"] = (
    emotion_stimuli_dict["ep01"]
    + emotion_stimuli_dict["ep02"]
    + emotion_stimuli_dict["ep03"]
)

"""
#### Initialize base QA Prompt
"""
logger.info("#### Initialize base QA Prompt")


qa_tmpl_str = """\
Context information is below.
---------------------
{{ context_str }}
---------------------
Given the context information and not prior knowledge, \
answer the query.
{{ emotion_str }}
Query: {{ query_str }}
Answer: \
"""
qa_tmpl = RichPromptTemplate(qa_tmpl_str)

"""
#### Prepend emotions
"""
logger.info("#### Prepend emotions")

QA_PROMPT_KEY = "response_synthesizer:text_qa_template"

async def run_and_evaluate(
    query_engine, eval_qa_pairs, batch_runner, emotion_stimuli_str, qa_tmpl
):
    """Run and evaluate."""
    new_qa_tmpl = qa_tmpl.partial_format(emotion_str=emotion_stimuli_str)

    old_qa_tmpl = query_engine.get_prompts()[QA_PROMPT_KEY]
    query_engine.update_prompts({QA_PROMPT_KEY: new_qa_tmpl})
    async def async_func_10():
        avg_correctness = await get_correctness(
            query_engine, eval_qa_pairs, batch_runner
        )
        return avg_correctness
    avg_correctness = asyncio.run(async_func_10())
    logger.success(format_json(avg_correctness))
    query_engine.update_prompts({QA_PROMPT_KEY: old_qa_tmpl})
    return avg_correctness

async def async_func_16():
    correctness_ep01 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep01"],
        qa_tmpl,
    )
    return correctness_ep01
correctness_ep01 = asyncio.run(async_func_16())
logger.success(format_json(correctness_ep01))

logger.debug(correctness_ep01)

async def async_func_26():
    correctness_ep02 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep02"],
        qa_tmpl,
    )
    return correctness_ep02
correctness_ep02 = asyncio.run(async_func_26())
logger.success(format_json(correctness_ep02))

logger.debug(correctness_ep02)

async def async_func_36():
    correctness_base = await run_and_evaluate(
        query_engine, eval_dataset.qr_pairs, batch_runner, "", qa_tmpl
    )
    return correctness_base
correctness_base = asyncio.run(async_func_36())
logger.success(format_json(correctness_base))

logger.debug(correctness_base)

"""
From this, we can see that more emotional prompts seem to lead to better performance!
"""
logger.info("From this, we can see that more emotional prompts seem to lead to better performance!")

logger.info("\n\n[DONE]", bright=True)