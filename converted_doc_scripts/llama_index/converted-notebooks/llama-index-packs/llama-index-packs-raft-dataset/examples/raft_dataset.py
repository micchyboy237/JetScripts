from jet.logger import CustomLogger
from llama_index.packs.raft_dataset import RAFTDatasetPack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# RAFT Dataset LlamaPack

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This LlamaPack implements RAFT: Adapting Language Model to Domain Specific RAG [paper](https://arxiv.org/abs/2403.10131)

Retrieval Augmented FineTuning (RAFT) is a training recipe introduced in this paper that aims to improve the performance of large language models (LLMs) in open-book, in-domain question-answering tasks. Given a question and a set of retrieved documents, RAFT trains the LLM to identify and cite verbatim the most relevant sequences from the documents that help answer the question, while ignoring irrelevant or distracting information. By explicitly training the model to distinguish between relevant and irrelevant information and to provide evidence from the relevant documents, RAFT encourages the LLM to develop better reasoning and explanation abilities, ultimately improving its ability to answer questions accurately and rationally in scenarios where additional context or knowledge is available.

A key component of RAFT is how the dataset is generated for fine-tuning. Each QA pair also includes an "oracle" document from which the answer to the question can be deduced as well as "distractor" documents which are irrelevant. During training this forces the model to learn which information is relevant/irrelevant and also memorize domain knowledge.

In this notebook we will create `RAFT Dataset` using `RAFTDatasetPack` LlamaPack.

#### Installation
"""
logger.info("# RAFT Dataset LlamaPack")

# !pip install llama-index
# !pip install llama-index-packs-raft-dataset


# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

"""
#### Download Data
"""
logger.info("#### Download Data")

# !wget --user-agent "Mozilla" "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" -O './paul_graham_essay.txt'


raft_dataset = RAFTDatasetPack("./paul_graham_essay.txt")

dataset = raft_dataset.run()

"""
The above dataset is HuggingFace Dataset format. You can then save it into `.arrow` or `.jsonl` format and use it for further finetuning.
"""
logger.info("The above dataset is HuggingFace Dataset format. You can then save it into `.arrow` or `.jsonl` format and use it for further finetuning.")

output_path = "<output path>"
dataset.save_to_disk(output_path)

dataset.to_json(output_path + ".jsonl")

"""
#### You can refer to the original implementation [here](https://github.com/ShishirPatil/gorilla/tree/main/raft)
"""
logger.info("#### You can refer to the original implementation [here](https://github.com/ShishirPatil/gorilla/tree/main/raft)")

logger.info("\n\n[DONE]", bright=True)