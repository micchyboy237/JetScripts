from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.embeddings import (
SelfHostedEmbeddings,
SelfHostedHuggingFaceEmbeddings,
SelfHostedHuggingFaceInstructEmbeddings,
)
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
pipeline,
)
import os
import runhouse as rh
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
# Self Hosted
Let's load the `SelfHostedEmbeddings`, `SelfHostedHuggingFaceEmbeddings`, and `SelfHostedHuggingFaceInstructEmbeddings` classes.
"""
logger.info("# Self Hosted")


gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

embeddings = SelfHostedHuggingFaceEmbeddings(hardware=gpu)

text = "This is a test document."

query_result = embeddings.embed_query(text)

"""
And similarly for SelfHostedHuggingFaceInstructEmbeddings:
"""
logger.info("And similarly for SelfHostedHuggingFaceInstructEmbeddings:")

embeddings = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)

"""
Now let's load an embedding model with a custom load function:
"""
logger.info("Now let's load an embedding model with a custom load function:")

def get_pipeline():

    model_id = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def inference_fn(pipeline, prompt):
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]

embeddings = SelfHostedEmbeddings(
    model_load_fn=get_pipeline,
    hardware=gpu,
    model_reqs=["./", "torch", "transformers"],
    inference_fn=inference_fn,
)

query_result = embeddings.embed_query(text)

logger.info("\n\n[DONE]", bright=True)