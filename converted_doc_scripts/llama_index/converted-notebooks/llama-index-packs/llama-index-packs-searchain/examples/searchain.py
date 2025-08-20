from jet.logger import CustomLogger
from llama_index.core.llama_pack import download_llama_pack
from searchain_pack.base import SearChainPack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# An Example of Searchain Application

This LlamaPack implements short form the [SearChain paper by Xu et al..](https://arxiv.org/abs/2304.14732)

This implementation is adapted from the author's implementation. You can find the official code repository [here](https://github.com/xsc1234/Search-in-the-Chain).

## Load Pack
"""
logger.info("# An Example of Searchain Application")

# ! pip install llama-index


download_llama_pack("SearChainPack", "./searchain_pack")

"""
## Setup
"""
logger.info("## Setup")

searchain = SearChainPack(
    data_path="data",
    dprtokenizer_path="./model/dpr_reader_multi",
    dprmodel_path="./model/dpr_reader_multi",
    crossencoder_name_or_path="./model/Quora_cross_encoder",
)

"""
## Excute
"""
logger.info("## Excute")

start_idx = 0
while not start_idx == -1:
    start_idx = execute(
        "/hotpotqa/hotpot_dev_fullwiki_v1_line.json", start_idx=start_idx
    )

logger.info("\n\n[DONE]", bright=True)