from jet.logger import logger
from langchain.chains.mapreduce import MapReduceChain
from langchain.model_laboratory import ModelLaboratory
from langchain_community.llms.manifest import ManifestWrapper
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from manifest import Manifest
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
# Manifest

This notebook goes over how to use Manifest and LangChain.

For more detailed information on `manifest`, and how to use it with local huggingface models like in this example, see https://github.com/HazyResearch/manifest

Another example of [using Manifest with Langchain](https://github.com/HazyResearch/manifest/blob/main/examples/langchain_chatgpt.html).
"""
logger.info("# Manifest")

# %pip install --upgrade --quiet  manifest-ml


manifest = Manifest(
    client_name="huggingface", client_connection="http://127.0.0.1:5000"
)
logger.debug(manifest.client_pool.get_current_client().get_model_params())

llm = ManifestWrapper(
    client=manifest, llm_kwargs={"temperature": 0.001, "max_tokens": 256}
)


_prompt = """Write a concise summary of the following:


{text}


CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(_prompt)

text_splitter = CharacterTextSplitter()

mp_chain = MapReduceChain.from_params(llm, prompt, text_splitter)

with open("../../how_to/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
mp_chain.run(state_of_the_union)

"""
## Compare HF Models
"""
logger.info("## Compare HF Models")


manifest1 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5000"
    ),
    llm_kwargs={"temperature": 0.01},
)
manifest2 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5001"
    ),
    llm_kwargs={"temperature": 0.01},
)
manifest3 = ManifestWrapper(
    client=Manifest(
        client_name="huggingface", client_connection="http://127.0.0.1:5002"
    ),
    llm_kwargs={"temperature": 0.01},
)
llms = [manifest1, manifest2, manifest3]
model_lab = ModelLaboratory(llms)

model_lab.compare("What color is a flamingo?")

logger.info("\n\n[DONE]", bright=True)