from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import GradientLLM
from langchain_core.prompts import PromptTemplate
import gradientai
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
# Gradient

`Gradient` allows to fine tune and get completions on LLMs with a simple web API.

This notebook goes over how to use Langchain with [Gradient](https://gradient.ai/).

## Imports
"""
logger.info("# Gradient")


"""
## Set the Environment API Key
Make sure to get your API key from Gradient AI. You are given $10 in free credits to test and fine-tune different models.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
#     os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
#     os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")

"""
Optional: Validate your Environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.
"""
logger.info("Optional: Validate your Environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.")

# %pip install --upgrade --quiet  gradientai


client = gradientai.Gradient()

models = client.list_models(only_base=True)
for model in models:
    logger.debug(model.id)

new_model = models[-1].create_model_adapter(name="my_model_adapter")
new_model.id, new_model.name

"""
## Create the Gradient instance
You can specify different parameters such as the model, max_tokens generated, temperature, etc.

As we later want to fine-tune out model, we select the model_adapter with the id `674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter`, but you can use any base or fine-tunable model.
"""
logger.info("## Create the Gradient instance")

llm = GradientLLM(
    model="674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter",
    model_kwargs=dict(max_generated_token_count=128),
)

"""
## Create a Prompt Template
We will create a prompt template for Question and Answer.
"""
logger.info("## Create a Prompt Template")

template = """Question: {question}

Answer: """

prompt = PromptTemplate.from_template(template)

"""
## Initiate the LLMChain
"""
logger.info("## Initiate the LLMChain")

llm_chain = LLMChain(prompt=prompt, llm=llm)

"""
## Run the LLMChain
Provide a question and run the LLMChain.
"""
logger.info("## Run the LLMChain")

question = "What NFL team won the Super Bowl in 1994?"

llm_chain.run(question=question)

"""
# Improve the results by fine-tuning (optional)
Well - that is wrong - the San Francisco 49ers did not win.
The correct answer to the question would be `The Dallas Cowboys!`.

Let's increase the odds for the correct answer, by fine-tuning on the correct answer using the PromptTemplate.
"""
logger.info("# Improve the results by fine-tuning (optional)")

dataset = [
    {
        "inputs": template.format(question="What NFL team won the Super Bowl in 1994?")
        + " The Dallas Cowboys!"
    }
]
dataset

new_model.fine_tune(samples=dataset)

llm_chain.run(question=question)

"""

"""

logger.info("\n\n[DONE]", bright=True)