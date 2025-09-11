from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import SelfHostedHuggingFaceLLM, SelfHostedPipeline
from langchain_core.prompts import PromptTemplate
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
pipeline,
)
import os
import pickle
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
# Runhouse

[Runhouse](https://github.com/run-house/runhouse) allows remote compute and data across environments and users. See the [Runhouse docs](https://www.run.house/docs).

This example goes over how to use LangChain and [Runhouse](https://github.com/run-house/runhouse) to interact with models hosted on your own GPU, or on-demand GPUs on AWS, GCP, AWS, or Lambda.

**Note**: Code uses `SelfHosted` name instead of the `Runhouse`.
"""
logger.info("# Runhouse")

# %pip install --upgrade --quiet  runhouse


gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = SelfHostedHuggingFaceLLM(
    model_id="gpt2", hardware=gpu, model_reqs=["pip:./", "transformers", "torch"]
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

"""
You can also load more custom models through the SelfHostedHuggingFaceLLM interface:
"""
logger.info("You can also load more custom models through the SelfHostedHuggingFaceLLM interface:")

llm = SelfHostedHuggingFaceLLM(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    hardware=gpu,
)

llm("What is the capital of Germany?")

"""
Using a custom load function, we can load a custom pipeline directly on the remote hardware:
"""
logger.info("Using a custom load function, we can load a custom pipeline directly on the remote hardware:")

def load_pipeline():

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    return pipe


def inference_fn(pipeline, prompt, stop=None):
    return pipeline(prompt)[0]["generated_text"][len(prompt) :]

llm = SelfHostedHuggingFaceLLM(
    model_load_fn=load_pipeline, hardware=gpu, inference_fn=inference_fn
)

llm("Who is the current US president?")

"""
You can send your pipeline directly over the wire to your model, but this will only work for small models (&lt;2 Gb), and will be pretty slow:
"""
logger.info("You can send your pipeline directly over the wire to your model, but this will only work for small models (&lt;2 Gb), and will be pretty slow:")

pipeline = load_pipeline()
llm = SelfHostedPipeline.from_pipeline(
    pipeline=pipeline, hardware=gpu, model_reqs=["pip:./", "transformers", "torch"]
)

"""
Instead, we can also send it to the hardware's filesystem, which will be much faster.
"""
logger.info("Instead, we can also send it to the hardware's filesystem, which will be much faster.")


rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(
    gpu, path="models"
)

llm = SelfHostedPipeline.from_pipeline(pipeline="models/pipeline.pkl", hardware=gpu)

logger.info("\n\n[DONE]", bright=True)