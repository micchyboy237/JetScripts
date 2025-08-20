from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
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
# Llama Pack - Resume Screener 📄

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llama_hub/llama_pack_resume.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This example shows you how to use the Resume Screener Llama Pack.
You can find all packs on https://llamahub.ai

The resume screener is designed to analyze a candidate's resume according to a set of criteria, and decide whether the candidate is a fit for the job.

in this example we'll evaluate a sample resume (e.g. Jerry's old resume).
"""
logger.info("# Llama Pack - Resume Screener 📄")

# %pip install llama-index-readers-wikipedia

# !pip install llama-index llama-hub

"""
### Setup Data

We'll load some sample Wikipedia data for MLX, Sam, Mira, and Emmett. Why? No reason in particular :)
"""
logger.info("### Setup Data")


loader = WikipediaReader()
documents = loader.load_data(
    pages=["MLX", "Sam Altman", "Mira Murati", "Emmett Shear"],
    auto_suggest=False,
)


sentence_splitter = SentenceSplitter(chunk_size=1024)

"""
We get the first chunk from each essay.
"""
logger.info("We get the first chunk from each essay.")

openai_node = sentence_splitter.get_nodes_from_documents([documents[0]])[0]
sama_node = sentence_splitter.get_nodes_from_documents([documents[1]])[0]
mira_node = sentence_splitter.get_nodes_from_documents([documents[2]])[0]
emmett_node = sentence_splitter.get_nodes_from_documents([documents[3]])[0]

"""
We'll also download Jerry's resume in 2019.

## Download Resume Screener Pack from LlamaHub

Here we download the resume screener pack class from LlamaHub.

We'll use it for two use cases:
- whether the candidate is a good fit for a front-end / full-stack engineering role.
- whether the candidate is a good fit for the CEO of MLX.
"""
logger.info("## Download Resume Screener Pack from LlamaHub")


ResumeScreenerPack = download_llama_pack(
    "ResumeScreenerPack", "./resume_screener_pack"
)

"""
### Screen Candidate for MLE Role

We take a job description on an MLE role from Meta's website.
"""
logger.info("### Screen Candidate for MLE Role")

meta_jd = """\
Meta is embarking on the most transformative change to its business and technology in company history, and our Machine Learning Engineers are at the forefront of this evolution. By leading crucial projects and initiatives that have never been done before, you have an opportunity to help us advance the way people connect around the world.

The ideal candidate will have industry experience working on a range of recommendation, classification, and optimization problems. You will bring the ability to own the whole ML life cycle, define projects and drive excellence across teams. You will work alongside the world’s leading engineers and researchers to solve some of the most exciting and massive social data and prediction problems that exist on the web.\
"""

resume_screener = ResumeScreenerPack(
    job_description=meta_jd,
    criteria=[
        "2+ years of experience in one or more of the following areas: machine learning, recommendation systems, pattern recognition, data mining, artificial intelligence, or related technical field",
        "Experience demonstrating technical leadership working with teams, owning projects, defining and setting technical direction for projects",
        "Bachelor's degree in Computer Science, Computer Engineering, relevant technical field, or equivalent practical experience.",
    ],
)

response = resume_screener.run(resume_path="jerry_resume.pdf")

for cd in response.criteria_decisions:
    logger.debug("### CRITERIA DECISION")
    logger.debug(cd.reasoning)
    logger.debug(cd.decision)
logger.debug("#### OVERALL REASONING ##### ")
logger.debug(str(response.overall_reasoning))
logger.debug(str(response.overall_decision))

"""
### Screen Candidate for FE / Typescript roles
"""
logger.info("### Screen Candidate for FE / Typescript roles")

resume_screener = ResumeScreenerPack(
    job_description="We're looking to hire a front-end engineer",
    criteria=[
        "The individual needs to be experienced in front-end / React / Typescript"
    ],
)

response = resume_screener.run(resume_path="jerry_resume.pdf")

logger.debug(str(response.overall_reasoning))
logger.debug(str(response.overall_decision))

"""
### Screen Candidate for CEO of MLX

Jerry can't write Typescript, but can he be CEO of MLX?
"""
logger.info("### Screen Candidate for CEO of MLX")

job_description = f"""\
We're looking to hire a CEO for MLX.

Instead of listing a set of specific criteria, each "criteria" is instead a short biography of a previous CEO.\

For each criteria/bio, outline if the candidate's experience matches or surpasses that of the candidate.

Also, here's a description of MLX from Wikipedia:
{openai_node.get_content()}
"""

profile_strs = [
    f"Profile: {n.get_content()}" for n in [sama_node, mira_node, emmett_node]
]


resume_screener = ResumeScreenerPack(
    job_description=job_description, criteria=profile_strs
)

response = resume_screener.run(resume_path="jerry_resume.pdf")

for cd in response.criteria_decisions:
    logger.debug("### CRITERIA DECISION")
    logger.debug(cd.reasoning)
    logger.debug(cd.decision)
logger.debug("#### OVERALL REASONING ##### ")
logger.debug(str(response.overall_reasoning))
logger.debug(str(response.overall_decision))

"""
...sadly not
"""

logger.info("\n\n[DONE]", bright=True)