import numpy as np
import pickle
from copy import deepcopy
from llama_index.core import PromptTemplate
from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.evaluation.eval_utils import get_responses
import random
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core import Settings
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
import os
import asyncio
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()


file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# "Optimization by Prompting" for RAG

Inspired by the [Optimization by Prompting paper](https://arxiv.org/pdf/2309.03409.pdf) by Yang et al., in this guide we test the ability of a "meta-prompt" to optimize our prompt for better RAG performance. The process is roughly as follows:
1. The prompt to be optimized is our standard QA prompt template for RAG, specifically the instruction prefix.
2. We have a "meta-prompt" that takes in previous prefixes/scores + an example of the task, and spits out another prefix.
3. For every candidate prefix, we compute a "score" through correctness evaluation - comparing a dataset of predicted answers (using the QA prompt) to a candidate dataset. If you don't have it already, you can generate with GPT-4.
"""

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file pymupdf

# import nest_asyncio

# nest_asyncio.apply()

"""
## Setup Data

We use the Llama 2 paper as the input data source for our RAG pipeline.
"""

# !mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"

# from llama_index.readers.file import PDFReader
# from llama_index.readers.file import PyMuPDFReader

# loader = PDFReader()
docs0 = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]


node_parser = SentenceSplitter(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)

"""
## Setup Vector Index over this Data

We load this data into an in-memory vector store (embedded with Ollama embeddings).

We'll be aggressively optimizing the QA prompt for this RAG pipeline.
"""


Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)

index = VectorStoreIndex(base_nodes)

query_engine = index.as_query_engine(similarity_top_k=2)

"""
## Get "Golden" Dataset

Here we generate a dataset of ground-truth QA pairs (or load it).

This will be used for two purposes: 
1) To generate some exemplars that we can put into the meta-prompt to illustrate the task
2) To generate an evaluation dataset to compute our objective score - so that the meta-prompt can try optimizing for this score.
"""


QUESTION_GEN_QUERY = f"""
You are a Job Employer. Your task is to setup multiple questions
for an upcoming interview. Each question should cover a part of the document. Generated questions should be complete so that its answers cover all context information provided.

Format response with numeric list of questions separated by newline.
Example response format:
1. Question 1
...continue

Output only the generated questions without any explanations.
"""

dataset_generator = DatasetGenerator(
    base_nodes[:20],
    llm=Ollama(model="llama3.1", request_timeout=300.0, context_window=4096),
    show_progress=True,
    question_gen_query=QUESTION_GEN_QUERY,
)


async def run_async_code_672f8f1d():
    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)
    return eval_dataset

eval_dataset = asyncio.run(run_async_code_672f8f1d())
logger.success(format_json(eval_dataset))

eval_dataset.save_json(f"{GENERATED_DIR}/llama2_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json(
    f"{GENERATED_DIR}/llama2_eval_qr_dataset.json"
)

"""
#### Get Dataset Samples
"""


full_qr_pairs = eval_dataset.qr_pairs

num_exemplars = 2
num_eval = 40
exemplar_qr_pairs = random.sample(full_qr_pairs, num_exemplars)

eval_qr_pairs = random.sample(full_qr_pairs, num_eval)

len(exemplar_qr_pairs)

"""
## Do Prompt Optimization

We now define the functions needed for prompt optimization. We first define an evaluator, and then we setup the meta-prompt which produces candidate instruction prefixes.

Finally we define and run the prompt optimization loop.
"""

"""
#### Get Evaluator
"""


evaluator_c = CorrectnessEvaluator(llm=Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096))
evaluator_dict = {
    "correctness": evaluator_c,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

"""
#### Define Correctness Eval Function
"""


async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

    async def async_func_5():
        eval_results = batch_runner.evaluate_responses(
            eval_qs, responses=pred_responses, reference=eval_answers
        )
        return eval_results
    eval_results = asyncio.run(async_func_5())
    logger.success(format_json(eval_results))
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness

"""
#### Initialize base QA Prompt
"""

QA_PROMPT_KEY = "response_synthesizer:text_qa_template"


llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

qa_tmpl_str = (
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

print(query_engine.get_prompts()[QA_PROMPT_KEY].get_template())

"""
#### Define Meta-Prompt
"""

meta_tmpl_str = """\
Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.
The score ranges from 1 to 5.

{prev_instruction_score_pairs}

Below we show the task. The <INS> tag is prepended to the below prompt template, e.g. as follows:

```
<INS>
{prompt_tmpl_str}
```

The prompt template contains template variables. Given an input set of template variables, the formatted prompt is then given to an LLM to get an output.

Some examples of template variable inputs and expected outputs are given below to illustrate the task. **NOTE**: These do NOT represent the \
entire evaluation dataset.

{qa_pairs_str}

We run every input in an evaluation dataset through an LLM. If the LLM-generated output doesn't match the expected output, we mark it as wrong (score 0).
A correct answer has a score of 1. The final "score" for an instruction is the average of scores across an evaluation dataset.
Write your new instruction (<INS>) that is different from the old ones and has a score as high as possible.

Instruction (<INS>): \
"""

meta_tmpl = PromptTemplate(meta_tmpl_str)

"""
#### Define Prompt Optimization Functions
"""


def format_meta_tmpl(
    prev_instr_score_pairs,
    prompt_tmpl_str,
    qa_pairs,
    meta_tmpl,
):
    """Call meta-prompt to generate new instruction."""
    pair_str_list = [
        f"Instruction (<INS>):\n{instr}\nScore:\n{score}"
        for instr, score in prev_instr_score_pairs
    ]
    full_instr_pair_str = "\n\n".join(pair_str_list)

    qa_str_list = [
        f"query_str:\n{query_str}\nAnswer:\n{answer}"
        for query_str, answer in qa_pairs
    ]
    full_qa_pair_str = "\n\n".join(qa_str_list)

    fmt_meta_tmpl = meta_tmpl.format(
        prev_instruction_score_pairs=full_instr_pair_str,
        prompt_tmpl_str=prompt_tmpl_str,
        qa_pairs_str=full_qa_pair_str,
    )
    return fmt_meta_tmpl


def get_full_prompt_template(cur_instr: str, prompt_tmpl):
    tmpl_str = prompt_tmpl.get_template()
    new_tmpl_str = cur_instr + "\n" + tmpl_str
    new_tmpl = PromptTemplate(new_tmpl_str)
    return new_tmpl


def _parse_meta_response(meta_response: str):
    return str(meta_response).split("\n")[0]


async def optimize_prompts(
    query_engine,
    initial_instr: str,
    base_prompt_tmpl,
    meta_tmpl,
    meta_llm,
    batch_eval_runner,
    eval_qa_pairs,
    exemplar_qa_pairs,
    num_iterations: int = 5,
):
    prev_instr_score_pairs = []
    base_prompt_tmpl_str = base_prompt_tmpl.get_template()

    cur_instr = initial_instr
    for idx in range(num_iterations):
        if idx > 0:
            fmt_meta_tmpl = format_meta_tmpl(
                prev_instr_score_pairs,
                base_prompt_tmpl_str,
                exemplar_qa_pairs,
                meta_tmpl,
            )
            meta_response = meta_llm.complete(fmt_meta_tmpl)
            print(fmt_meta_tmpl)
            print(str(meta_response))
            cur_instr = _parse_meta_response(meta_response)

        new_prompt_tmpl = get_full_prompt_template(cur_instr, base_prompt_tmpl)
        query_engine.update_prompts({QA_PROMPT_KEY: new_prompt_tmpl})

        async def async_func_38():
            avg_correctness = await get_correctness(
                query_engine, eval_qa_pairs, batch_runner
            )
            return avg_correctness
        avg_correctness = asyncio.run(async_func_38())
        logger.success(format_json(avg_correctness))
        prev_instr_score_pairs.append((cur_instr, avg_correctness))

    max_instr_score_pair = max(
        prev_instr_score_pairs, key=lambda item: item[1]
    )

    return max_instr_score_pair[0], prev_instr_score_pairs

query_engine = index.as_query_engine(similarity_top_k=2)

base_qa_prompt = query_engine.get_prompts()[QA_PROMPT_KEY]


initial_instr = """\
You are a QA assistant.
Context information is below. Given the context information and not prior knowledge, \
answer the query. \
"""

old_qa_prompt = get_full_prompt_template(initial_instr, base_qa_prompt)

meta_llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)


async def async_func_0():
    new_instr, prev_instr_score_pairs = await optimize_prompts(
        query_engine,
        initial_instr,
        base_qa_prompt,
        meta_tmpl,
        meta_llm,  # note: treat llm as meta_llm
        batch_runner,
        eval_qr_pairs,
        exemplar_qr_pairs,
        num_iterations=5,
    )
    return new_instr, prev_instr_score_pairs
new_instr, prev_instr_score_pairs = asyncio.run(async_func_0())
logger.success(format_json(new_instr, prev_instr_score_pairs))


new_qa_prompt = query_engine.get_prompts()[QA_PROMPT_KEY]
print(new_qa_prompt)


pickle.dump(prev_instr_score_pairs, open("prev_instr_score_pairs.pkl", "wb"))

prev_instr_score_pairs

full_eval_qs = [q for q, _ in full_qr_pairs]
full_eval_answers = [a for _, a in full_qr_pairs]

query_engine.update_prompts({QA_PROMPT_KEY: old_qa_prompt})


async def async_func_1():
    avg_correctness_old = await get_correctness(
        query_engine, full_qr_pairs, batch_runner
    )
    return avg_correctness_old
avg_correctness_old = asyncio.run(async_func_1())
logger.success(format_json(avg_correctness_old))

print(avg_correctness_old)

query_engine.update_prompts({QA_PROMPT_KEY: new_qa_prompt})


async def async_func_1():
    avg_correctness_new = await get_correctness(
        query_engine, full_qr_pairs, batch_runner
    )
    return avg_correctness_new
avg_correctness_new = asyncio.run(async_func_1())
logger.success(format_json(avg_correctness_new))

print(avg_correctness_new)

logger.info("\n\n[DONE]", bright=True)
