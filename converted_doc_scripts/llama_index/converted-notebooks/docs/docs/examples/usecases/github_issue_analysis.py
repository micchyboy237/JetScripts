import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.async_utils import batch_gather
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.program.openai import MLXPydanticProgram
from llama_index.readers.github import (
GitHubRepositoryIssuesReader,
GitHubIssuesClient,
)
from pydantic import BaseModel
from tqdm.asyncio import asyncio
from typing import List
import os
import pickle
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
# Github Issue Analysis

## Setup

To use the github repo issue loader, you need to set your github token in the environment.  

See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for how to get a github token.  
See [llama-hub](https://llama-hub-ui.vercel.app/l/github_repo_issues) for more details about the loader.
"""
logger.info("# Github Issue Analysis")

# %pip install llama-index-readers-github
# %pip install llama-index-llms-ollama
# %pip install llama-index-program-openai


os.environ["GITHUB_TOKEN"] = "<your github token>"

"""
## Load Github Issue tickets
"""
logger.info("## Load Github Issue tickets")



github_client = GitHubIssuesClient()
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="jerryjliu",
    repo="llama_index",
    verbose=True,
)

docs = loader.load_data()

"""
Quick inspection
"""
logger.info("Quick inspection")

docs[10].text

docs[10].metadata

"""
## Extract themes
"""
logger.info("## Extract themes")

# %load_ext autoreload
# %autoreload 2




prompt_template_str = """\
Here is a Github Issue ticket.

{ticket}

Please extract central themes and output a list of tags.\
"""

class TagList(BaseModel):
    """A list of tags corresponding to central themes of an issue."""

    tags: List[str]

program = MLXPydanticProgram.from_defaults(
    prompt_template_str=prompt_template_str,
    output_cls=TagList,
)

tasks = [program.acall(ticket=doc) for doc in docs]

async def run_async_code_b6259ecd():
    async def run_async_code_a54895ec():
        output = await batch_gather(tasks, batch_size=10, verbose=True)
        return output
    output = asyncio.run(run_async_code_a54895ec())
    logger.success(format_json(output))
    return output
output = asyncio.run(run_async_code_b6259ecd())
logger.success(format_json(output))

"""
## [Optional] Save/Load Extracted Themes
"""
logger.info("## [Optional] Save/Load Extracted Themes")


with open("github_issue_analysis_data.pkl", "wb") as f:
    pickle.dump(tag_lists, f)

with open("github_issue_analysis_data.pkl", "rb") as f:
    tag_lists = pickle.load(f)
    logger.debug(f"Loaded tag lists for {len(tag_lists)} tickets")

"""
## Summarize Themes

Build prompt
"""
logger.info("## Summarize Themes")

prompt = """
Here is a list of central themes (in the form of tags) extracted from a list of Github Issue tickets.
Tags for each ticket is separated by 2 newlines.

{tag_lists_str}

Please summarize the key takeaways and what we should prioritize to fix.
"""

tag_lists_str = "\n\n".join([str(tag_list) for tag_list in tag_lists])

prompt = prompt.format(tag_lists_str=tag_lists_str)

"""
Summarize with GPT-4
"""
logger.info("Summarize with GPT-4")


response = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats").stream_complete(prompt)

for r in response:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)