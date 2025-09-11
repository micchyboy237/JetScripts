from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.callbacks import ClearMLCallbackHandler
from langchain_core.callbacks import StdOutCallbackHandler
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
# ClearML

> [ClearML](https://github.com/allegroai/clearml) is a ML/DL development and production suite, it contains 5 main modules:
> - `Experiment Manager` - Automagical experiment tracking, environments and results
> - `MLOps` - Orchestration, Automation & Pipelines solution for ML/DL jobs (K8s / Cloud / bare-metal)
> - `Data-Management` - Fully differentiable data management & version control solution on top of object-storage (S3 / GS / Azure / NAS)
> - `Model-Serving` - cloud-ready Scalable model serving solution!
    Deploy new model endpoints in under 5 minutes
    Includes optimized GPU serving support backed by Nvidia-Triton
    with out-of-the-box Model Monitoring
> - `Fire Reports` - Create and share rich MarkDown documents supporting embeddable online content

In order to properly keep track of your langchain experiments and their results, you can enable the `ClearML` integration. We use the `ClearML Experiment Manager` that neatly tracks and organizes all your experiment runs.

<a target="_blank" href="https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/clearml_tracking.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Installation and Setup
"""
logger.info("# ClearML")

# %pip install --upgrade --quiet  clearml
# %pip install --upgrade --quiet  pandas
# %pip install --upgrade --quiet  textstat
# %pip install --upgrade --quiet  spacy
# !python -m spacy download en_core_web_sm

"""
### Getting API Credentials

We'll be using quite some APIs in this notebook, here is a list and where to get them:

- ClearML: https://app.clear.ml/settings/workspace-configuration
- Ollama: https://platform.ollama.com/account/api-keys
- SerpAPI (google search): https://serpapi.com/dashboard
"""
logger.info("### Getting API Credentials")


os.environ["CLEARML_API_ACCESS_KEY"] = ""
os.environ["CLEARML_API_SECRET_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

"""
## Callbacks
"""
logger.info("## Callbacks")


clearml_callback = ClearMLCallbackHandler(
    task_type="inference",
    project_name="langchain_callback_demo",
    task_name="llm",
    tags=["test"],
    visualize=True,
    complexity_metrics=True,
    stream_logs=True,
)
callbacks = [StdOutCallbackHandler(), clearml_callback]
llm = ChatOllama(temperature=0, callbacks=callbacks)

"""
### Scenario 1: Just an LLM

First, let's just run a single LLM a few times and capture the resulting prompt-answer conversation in ClearML
"""
logger.info("### Scenario 1: Just an LLM")

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 3)
clearml_callback.flush_tracker(langchain_asset=llm, name="simple_sequential")

"""
At this point you can already go to https://app.clear.ml and take a look at the resulting ClearML Task that was created.

Among others, you should see that this notebook is saved along with any git information. The model JSON that contains the used parameters is saved as an artifact, there are also console logs and under the plots section, you'll find tables that represent the flow of the chain.

Finally, if you enabled visualizations, these are stored as HTML files under debug samples.

### Scenario 2: Creating an agent with tools

To show a more advanced workflow, let's create an agent with access to tools. The way ClearML tracks the results is not different though, only the table will look slightly different as there are other types of actions taken when compared to the earlier, simpler example.

You can now also see the use of the `finish=True` keyword, which will fully close the ClearML Task, instead of just resetting the parameters and prompts for a new conversation.
"""
logger.info("### Scenario 2: Creating an agent with tools")


tools = load_tools(["serpapi", "llm-math"], llm=llm, callbacks=callbacks)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=callbacks,
)
agent.run("Who is the wife of the person who sang summer of 69?")
clearml_callback.flush_tracker(
    langchain_asset=agent, name="Agent with Tools", finish=True
)

"""
### Tips and Next Steps

- Make sure you always use a unique `name` argument for the `clearml_callback.flush_tracker` function. If not, the model parameters used for a run will override the previous run!

- If you close the ClearML Callback using `clearml_callback.flush_tracker(..., finish=True)` the Callback cannot be used anymore. Make a new one if you want to keep logging.

- Check out the rest of the open-source ClearML ecosystem, there is a data version manager, a remote execution agent, automated pipelines and much more!
"""
logger.info("### Tips and Next Steps")


logger.info("\n\n[DONE]", bright=True)
