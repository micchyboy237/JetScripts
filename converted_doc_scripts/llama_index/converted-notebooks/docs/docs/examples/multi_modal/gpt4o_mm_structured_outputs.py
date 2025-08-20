from IPython.display import clear_output
from PIL import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import MLXMultiModal
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
import tqdm


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/gpt4o_mm_structured_outputs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multimodal Structured Outputs: GPT-4o vs. Other GPT-4 Variants

In this notebook, we use the `MultiModalLLMCompletionProgram` class to perform structured data extraction with images. We'll make comparisons across the the GPT-4 vision-capable models.
"""
logger.info("# Multimodal Structured Outputs: GPT-4o vs. Other GPT-4 Variants")

# %pip install llama-index-llms-ollama -q
# %pip install llama-index-multi-modal-llms-openai -q
# %pip install llama-index-readers-file -q
# %pip install -U llama-index-core -q


"""
## The Image Dataset: PaperCards

For this data extraction task, we'll be using the multimodal LLMs to extract information from so-called PaperCards. These are visualizations containing summaries of research papers. The dataset can be downloaded from our dropbox account by executing the command below.

### Download the images
"""
logger.info("## The Image Dataset: PaperCards")

# !mkdir data
# !wget "https://www.dropbox.com/scl/fo/jlxavjjzddcv6owvr9e6y/AJoNd0T2pUSeynOTtM_f60c?rlkey=4mvwc1r6lowmy7zqpnm1ikd24&st=1cs1gs9c&dl=1" -O data/paper_cards.zip
# !unzip data/paper_cards.zip -d data
# !rm data/paper_cards.zip

"""
### Load PaperCards as ImageDocuments
"""
logger.info("### Load PaperCards as ImageDocuments")


image_path = "./data"
image_documents = SimpleDirectoryReader(image_path).load_data()

img_doc = image_documents[0]
image = Image.open(img_doc.image_path).convert("RGB")
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(image)
plt.show()

"""
## Build Our MultiModalLLMCompletionProgram (Multimodal Structured Outputs)

### Desired Structured Output

Here we will define our data class (i.e., Pydantic BaseModel) that will hold the data that we extract from a given image or PaperCard.
"""
logger.info("## Build Our MultiModalLLMCompletionProgram (Multimodal Structured Outputs)")



class PaperCard(BaseModel):
    """Data class for storing text attributes of a PaperCard."""

    title: str = Field(description="Title of paper.")
    year: str = Field(description="Year of publication of paper.")
    authors: str = Field(description="Authors of paper.")
    arxiv_id: str = Field(description="Arxiv paper id.")
    main_contribution: str = Field(
        description="Main contribution of the paper."
    )
    insights: str = Field(
        description="Main insight or motivation for the paper."
    )
    main_results: List[str] = Field(
        description="The main results of the paper."
    )
    tech_bits: Optional[str] = Field(
        description="Describe what's being displayed in the technical bits section of the image."
    )

"""
Next, we define our `MultiModalLLMCompletionProgram`. Here we actually will define three separate programs, one for each of the vision-capable GPT-4 models, namely: GPT-4o, GPT-4v, and GPT-4Turbo.
"""
logger.info("Next, we define our `MultiModalLLMCompletionProgram`. Here we actually will define three separate programs, one for each of the vision-capable GPT-4 models, namely: GPT-4o, GPT-4v, and GPT-4Turbo.")

paper_card_extraction_prompt = """
Use the attached PaperCard image to extract data from it and store into the
provided data class.
"""

gpt_4o = MLXMultiModal(model="qwen3-1.7b-4bit", max_new_tokens=4096)

gpt_4v = MLXMultiModal(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_new_tokens=4096)

gpt_4turbo = MLXMultiModal(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_new_tokens=4096
)

multimodal_llms = {
    "gpt_4o": gpt_4o,
    "gpt_4v": gpt_4v,
    "gpt_4turbo": gpt_4turbo,
}

programs = {
    mdl_name: MultiModalLLMCompletionProgram.from_defaults(
        output_cls=PaperCard,
        prompt_template_str=paper_card_extraction_prompt,
        multi_modal_llm=mdl,
    )
    for mdl_name, mdl in multimodal_llms.items()
}

"""
### Let's give it a test run
"""
logger.info("### Let's give it a test run")

papercard = programs["gpt_4o"](image_documents=[image_documents[0]])

papercard

"""
## Run The Data Extraction Task

Now that we've tested our program, we're ready to apply the programs to the data extraction task over the PaperCards!
"""
logger.info("## Run The Data Extraction Task")


results = {}

for mdl_name, program in programs.items():
    logger.debug(f"Model: {mdl_name}")
    results[mdl_name] = {
        "papercards": [],
        "failures": [],
        "execution_times": [],
        "image_paths": [],
    }
    total_time = 0
    for img in tqdm.tqdm(image_documents):
        results[mdl_name]["image_paths"].append(img.image_path)
        start_time = time.time()
        try:
            structured_output = program(image_documents=[img])
            end_time = time.time() - start_time
            results[mdl_name]["papercards"].append(structured_output)
            results[mdl_name]["execution_times"].append(end_time)
            results[mdl_name]["failures"].append(None)
        except Exception as e:
            results[mdl_name]["papercards"].append(None)
            results[mdl_name]["execution_times"].append(None)
            results[mdl_name]["failures"].append(e)
    logger.debug()

"""
## Quantitative Analysis

Here, we'll perform a quick quantitative analysis of the various programs. Specifically, we compare the total number of failures, total execution time of successful data extraction jobs, and the average execution time.
"""
logger.info("## Quantitative Analysis")


metrics = {
    "gpt_4o": {},
    "gpt_4v": {},
    "gpt_4turbo": {},
}

for mdl_name, mdl_results in results.items():
    metrics[mdl_name]["error_count"] = sum(
        el is not None for el in mdl_results["failures"]
    )
    metrics[mdl_name]["total_execution_time"] = sum(
        el for el in mdl_results["execution_times"] if el is not None
    )
    metrics[mdl_name]["average_execution_time"] = metrics[mdl_name][
        "total_execution_time"
    ] / (len(image_documents) - metrics[mdl_name]["error_count"])
    metrics[mdl_name]["median_execution_time"] = np.percentile(
        [el for el in mdl_results["execution_times"] if el is not None], q=0.5
    )

pd.DataFrame(metrics)

"""
### GPT-4o is indeed faster!

- GPT-4o is clearly faster in both total execution time (of successful programs, failed extractions are not counted here) as well as mean and median execution times
- Not only is GPT-4o faster, it was able to yield an extraction for all PaperCards. In contrast, GPT-4v failed 14 times, and GPT-4turbo failed 1 time.

## Qualitative Analysis

In this final section, we'll conduct a qualitative analysis over the extraction results. Ultimately, we'll end up with a "labelled" dataset of human evaluations on the data extraction task. The utilities provided next will allow you to perform a manual evaluation on the results of the three programs (or models) per PaperCard data extraction. Your job as a labeller is to rank the program's result from 0 to 5 with 5 being a perfect data extraction.
"""
logger.info("### GPT-4o is indeed faster!")


def display_results_and_papercard(ix: int):
    image_path = results["gpt_4o"]["image_paths"][ix]

    gpt_4o_output = results["gpt_4o"]["papercards"][ix]
    gpt_4v_output = results["gpt_4v"]["papercards"][ix]
    gpt_4turbo_output = results["gpt_4turbo"]["papercards"][ix]

    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

    logger.debug("GPT-4o\n")
    if gpt_4o_output is not None:
        logger.debug(json.dumps(gpt_4o_output.dict(), indent=4))
    else:
        logger.debug("Failed to extract data")
    logger.debug()
    logger.debug("============================================\n")

    logger.debug("GPT-4v\n")
    if gpt_4v_output is not None:
        logger.debug(json.dumps(gpt_4v_output.dict(), indent=4))
    else:
        logger.debug("Failed to extract data")
    logger.debug()
    logger.debug("============================================\n")

    logger.debug("GPT-4turbo\n")
    if gpt_4turbo_output is not None:
        logger.debug(json.dumps(gpt_4turbo_output.dict(), indent=4))
    else:
        logger.debug("Failed to extract data")
    logger.debug()
    logger.debug("============================================\n")

GRADES = {
    "gpt_4o": [0] * len(image_documents),
    "gpt_4v": [0] * len(image_documents),
    "gpt_4turbo": [0] * len(image_documents),
}


def manual_evaluation_single(img_ix: int):
    """Update the GRADES dictionary for a single PaperCard
    data extraction task.
    """
    display_results_and_papercard(img_ix)

    gpt_4o_grade = input(
        "Provide a rating from 0 to 5, with 5 being the highest for GPT-4o."
    )
    gpt_4v_grade = input(
        "Provide a rating from 0 to 5, with 5 being the highest for GPT-4v."
    )
    gpt_4turbo_grade = input(
        "Provide a rating from 0 to 5, with 5 being the highest for GPT-4turbo."
    )

    GRADES["gpt_4o"][img_ix] = gpt_4o_grade
    GRADES["gpt_4v"][img_ix] = gpt_4v_grade
    GRADES["gpt_4turbo"][img_ix] = gpt_4turbo_grade


def manual_evaluations(img_ix: Optional[int] = None):
    """An interactive program for manually grading gpt-4 variants on the
    task of PaperCard data extraction.
    """
    if img_ix is None:
        for ix in range(len(image_documents)):
            logger.debug(f"You are marking {ix + 1} out of {len(image_documents)}")
            logger.debug()
            manual_evaluation_single(ix)
            clear_output(wait=True)
    else:
        manual_evaluation_single(img_ix)

manual_evaluations()

grades_df = pd.DataFrame(GRADES, dtype=float)
grades_df.mean()

"""
### Table Of Observations

In the table below, we list our general observations per component that we wished to extract from the PaperCard. GPT-4v and and GPT-4Turbo performed similarly with a slight edge to GPT-4Turbo. Generally speaking, GPT-4o demonstrated significantly better performance in this data extraction task than the other models. Finally, all models seemed to struggle on describing the Tech Bits section of the PaperCard, and at times, all of the models would generate a summary instead of an exact extraction; however, GPT-4o did this less than the other models.

| Extracted component  	| GPT-4o                                                         	| GPT-4v & GPT-4Turbo                                       	|
| :- | :- | :- |
| Title, Year, Authors 	| very good, probably 100%                                       	| probably 80%, hallucinated on few examples                	|
| Arxiv ID             	| good, around 95% accurate                                      	| 70% accurate                                              	|
| Main Contribution    	| good (~80%) but couldn't extract multiple contributions listed 	| not so great, 60% accurate, some halluciations            	|
| Insights             	| not so good (~65%) did more summarization then extraction      	| did more summarization then extraction                    	|
| Main Results         	| very good at extracting summary statements of main results     	| hallucinated a lot here                                   	|
| Tech Bits            	| unable to generate detailed descriptions of diagrams here      	| unable to generate detailed descriptions of diagrams here 	|

## Summary

- GPT-4o is faster and fails less (0 times!) than GPT-4v and GPT-4turbo
- GPT-4o yields better data extraction results than GPT-4v and GPT-4turbo
- GPT-4o was very good at extracting facts from the PaperCard: Title, Author, Year, and headline statements of the Main Results section
- GPT-4v and GPT-4turbo often hallucinated the main results and sometimes the authors
- Results with GPT-4o can probably be improved using better prompting especially for extracting data from Insights section, but also for describing Tech Bits
"""
logger.info("### Table Of Observations")

logger.info("\n\n[DONE]", bright=True)