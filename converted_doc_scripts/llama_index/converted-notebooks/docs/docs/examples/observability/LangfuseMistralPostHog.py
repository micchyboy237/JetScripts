from jet.logger import CustomLogger
from langfuse import get_client
from langfuse import observe, get_client
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import os
import shutil
import wget


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/LangfuseMistralPostHog.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Analyze and Debug LlamaIndex Applications with PostHog and Langfuse

In this cookbook, we show you how to build a RAG application with [LlamaIndex](https://www.llamaindex.ai/), observe the steps with [Langfuse](https://langfuse.com/), and analyze the data in [PostHog](https://posthog.com/).

## What is Langfuse?
[Langfuse](https://langfuse.com/) is an open-source LLM engineering platform designed to help engineers understand and optimize user interactions with their language model applications. It provides tools for tracking, debugging, and improving LLM performance in real-world use cases. Langfuse is available both as a managed [cloud solution](https://cloud.langfuse.com/) and for [local or self-hosted](https://langfuse.com/docs/deployment/feature-overview) deployments.

## What is PostHog?
[PostHog](https://posthog.com/) is a popular choice for product analytics. Combining Langfuse's LLM analytics with PostHog's product analytics makes it easy to:

- **Analyze User Engagement**: Determine how often users interact with specific LLM features and understand their overall activity patterns.
- **Correlate Feedback with Behavior**: See how user feedback captured in Langfuse correlates with user behavior in PostHog.
- **Monitor LLM Performance**: Track and analyze metrics such as model cost, latency, and user feedback to optimize LLM performance.

## What is LlamaIndex?
LlamaIndex [(GitHub)](https://github.com/run-llama/llama_index) is a data framework designed to connect LLMs with external data sources. It helps structure, index, and query data effectively. This makes it easier for developers to build advanced LLM applications.

## How to Build a Simple RAG Application with LlamaIndex and Mistral

In this tutorial, we demonstrate how to create a chat application that provides answers to questions about hedgehog care. LlamaIndex is used to vectorize a [hedgehog care guide](https://www.pro-igel.de/downloads/merkblaetter_engl/wildtier_engl.pdf) with the [Mistral 8x22B model](https://docs.mistral.ai/getting-started/models/). All model generations are then traced using Langfuse's [LlamaIndex integration](https://langfuse.com/docs/integrations/llama-index/get-started).

Finally, the [PostHog integration](https://langfuse.com/docs/analytics/posthog) allows you to view detailed analytics about your hedgehog application directly in PostHog.

### Step 1: Set up LlamaIndex and Mistral

First, we set our Mistral API key as an environment variable. If you haven't already, [sign up for a Mistral acccount](https://console.mistral.ai/). Then [subscribe](https://console.mistral.ai/billing/) to a free trial or billing plan, after which you'll be able to [generate an API key](https://console.mistral.ai/api-keys/) (💡 You can use any other model supported by LlamaIndex; we just use Mistral in this cookbook).

Then, we use LlamaIndex to initialize both a Mistral language model and an embedding model. We then set these models in the LlamaIndex `Settings` object:
"""
logger.info("# Analyze and Debug LlamaIndex Applications with PostHog and Langfuse")

# %pip install llama-index llama-index-llms-mistralai llama-index-embeddings-mistralai nest_asyncio --upgrade


os.environ["MISTRAL_API_KEY"] = "***"

# import nest_asyncio

# nest_asyncio.apply()


llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model

"""
### Step 2: Initialize Langfuse

Next, we initialize the Langfuse client. [Sign up](https://cloud.langfuse.com/auth/sign-up) for Langfuse if you haven't already. Copy your [API keys](https://langfuse.com/faq/all/where-are-langfuse-api-keys) from your project settings and add them to your environment.
"""
logger.info("### Step 2: Initialize Langfuse")

# %pip install langfuse openinference-instrumentation-llama-index wget


os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # 🇪🇺 EU region

"""
With the environment variables set, we can now initialize the Langfuse client. get_client() initializes the Langfuse client using the credentials provided in the environment variables.
"""
logger.info("With the environment variables set, we can now initialize the Langfuse client. get_client() initializes the Langfuse client using the credentials provided in the environment variables.")


langfuse = get_client()

if langfuse.auth_check():
    logger.debug("Langfuse client is authenticated and ready!")
else:
    logger.debug("Authentication failed. Please check your credentials and host.")

"""
Now, we initialize the [OpenInference LlamaIndex instrumentation](https://docs.arize.com/phoenix/tracing/integrations-tracing/llamaindex). This third-party instrumentation automatically captures LlamaIndex operations and exports OpenTelemetry (OTel) spans to Langfuse.

Find out more about the Langfuse's LlamaIndex integration [here](https://langfuse.com/docs/integrations/llama-index/get-started).
"""
logger.info("Now, we initialize the [OpenInference LlamaIndex instrumentation](https://docs.arize.com/phoenix/tracing/integrations-tracing/llamaindex). This third-party instrumentation automatically captures LlamaIndex operations and exports OpenTelemetry (OTel) spans to Langfuse.")


LlamaIndexInstrumentor().instrument()

"""
### Step 3: Download data

We download the file we want to use for RAG. In this example, we use a [hedgehog care guide](https://www.pro-igel.de/downloads/merkblaetter_engl/wildtier_engl.pdf) pdf file to enable the language model to answer questions about caring for hedgehogs 🦔.
"""
logger.info("### Step 3: Download data")


url = "https://www.pro-igel.de/downloads/merkblaetter_engl/wildtier_engl.pdf"
wget.download(url, "./hedgehog.pdf")  # saves as ./hedgehog.pdf

"""
Next, we load the pdf using the LlamaIndex [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/).
"""
logger.info("Next, we load the pdf using the LlamaIndex [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/).")


hedgehog_docs = SimpleDirectoryReader(
    input_files=["./hedgehog.pdf"]
).load_data()

"""
### Step 4: Build RAG on the hedgehog doc

Next, we create vector embeddings of the hedgehog document using [`VectorStoreIndex`](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/) and then convert it into a [queryable engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/) to retrieve information based on queries.
"""
logger.info("### Step 4: Build RAG on the hedgehog doc")


hedgehog_index = VectorStoreIndex.from_documents(hedgehog_docs)
hedgehog_query_engine = hedgehog_index.as_query_engine(similarity_top_k=5)

"""
Finally, to put everything together, we query the engine and print a response:
"""
logger.info("Finally, to put everything together, we query the engine and print a response:")

response = hedgehog_query_engine.query("Which hedgehogs require help?")
logger.debug(response)

"""
All steps of the LLM chain are now tracked in Langfuse.

Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/367db23d-5b03-446b-bc73-36e289596c00

![Example trace in the Langfuse UI](https://langfuse.com/images/cookbook/example-posthog-llamaindex-mistral/trace-posthog-llamaindex-miostral.png)

### Step 5: (Optional) Implement user feedback to see how your application is performing

To monitor the quality of your hedgehog chat application, you can use [Langfuse Scores](https://langfuse.com/docs/scores/overview) to store user feedback (e.g. thumps up/down or comments). These scores can then be analysed in PostHog.

Scores are used to evaluate single observations or entire traces. You can create them via the annotation workflow in the Langfuse UI, run model-based evaluation or ingest via the SDK as we do it in this example.

To get the context of the current observation, we use the [`observe()` decorator](https://langfuse.com/docs/sdk/python/decorators) and apply it to the hedgehog_helper() function.
"""
logger.info("### Step 5: (Optional) Implement user feedback to see how your application is performing")


langfuse = get_client()


@observe()
def hedgehog_helper(user_message):
    response = hedgehog_query_engine.query(user_message)
    trace_id = langfuse.get_current_trace_id()

    logger.debug(response)

    return trace_id


trace_id = hedgehog_helper("Can I keep the hedgehog as a pet?")

langfuse.create_score(
    trace_id=trace_id,
    name="user-explicit-feedback",
    value=0.9,
    data_type="NUMERIC",  # optional, inferred if not provided
    comment="Good to know!",  # optional
)

"""
### Step 6: See your data in PostHog

Finally, we connect PostHog to our Langfuse account. Below is a summary of the steps to take (or see the [docs](https://posthog.com/docs/ai-engineering/langfuse-posthog) for full details):

1. [Sign up](https://us.posthog.com/) for your free PostHog account if you haven't already
2. Copy both your project API key and host from your [project settings](https://us.posthog.com/project/settings/project-details).
3. In your [Langfuse dashboard](https://cloud.langfuse.com/), click on **Settings** and scroll down to the **Integrations** section to find the PostHog integration.
4. Click **Configure** and paste in your PostHog host and project API key (you can find these in your [PostHog project settings](https://us.posthog.com/settings/project)).
5. Click **Enabled** and then **Save**.

 Langfuse will then begin exporting your data to PostHog once a day.

**Using the Langfuse dashboard template:**

Once you've installed the integration, [dashboard templates](https://posthog.com/docs/ai-engineering/langfuse-posthog#using-the-langfuse-dashboard-template) help you quickly set up relevant insights.

For our hedgehog chat application, we are using the template dashboard shown below. This enables you to analyze model cost, user feedback, and latency in PostHog.

To create your own dashboard from a template:

1. Go to the [dashboard](https://us.posthog.com/dashboard) tab in PostHog.
2. Click the **New dashboard** button in the top right.
3. Select **LLM metrics – Langfuse** from the list of templates.

![Posthog Dashboard showing user feedback and number of traces](https://langfuse.com/images/cookbook/example-posthog-llamaindex-mistral/dashboard-posthog-1.png)

![Posthog Dashboard showing latency and costs](https://langfuse.com/images/cookbook/example-posthog-llamaindex-mistral/dashboard-posthog-2.png)

## Feedback

If you have any feedback or requests, please create a GitHub [Issue](https://github.com/orgs/langfuse/discussions) or share your idea with the community on [Discord](https://discord.langfuse.com/).
"""
logger.info("### Step 6: See your data in PostHog")

logger.info("\n\n[DONE]", bright=True)